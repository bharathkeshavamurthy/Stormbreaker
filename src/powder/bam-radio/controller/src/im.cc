//  Copyright © 2018 Stephen Larew

#include "im.h"
#include "events.h"
#include <queue>
#include <valarray>

#include <boost/format.hpp>

using boost::format;
using namespace std::chrono_literals;
namespace chrono = std::chrono;

typedef std::chrono::duration<float, chrono::seconds::period> fsec;

namespace bamradio {

std::string LinkInfo::description() const {
  return (boost::format("%f s frame %lu sbo %f bps %f g bps %f ag bps") %
          frame_overhead.count() % segment_bit_overhead % throughput_bps %
          goodput_bps % goodput_arq_bps)
      .str();
}

std::string IndividualMandate::description() const {
  return visit<std::string>(
      [](StreamPT const &streampt) {
        return (boost::format("%f min bps %f s max latency") %
                streampt.min_throughput_bps % streampt.max_latency.count())
            .str();
      },
      [](FilePT const &filept) {
        return (boost::format("%f s file transfer duration %ld bytes") %
                filept.transfer_duration.count() % filept.size_bytes)
            .str();
      });
}

/// Adding new ranking and branching metrics for prioritized flow scheduling
QuantumSchedule scheduleFlows(std::vector< std::pair< FlowID, std::pair< float, FlowInfo > > > &reorderingVector,
                              std::map<FlowUID, FileFlowProgress> const &ffp,
                              chrono::system_clock::time_point now) {
  // An initial validation check
  if (reorderingVector.empty()) {
    return QuantumSchedule{{}, true, 0s, 0s, 0s};
  }
  auto const Nf = reorderingVector.size();
  // Compute min of max_latency and compute min time quantums.
  // Amax = min_{i in streams} stream_i.alpha * stream_i.max_latency
  // Amin_i = link.frame_overhead
  //            + (link.segment_bit_overhead + bitsPerSegment) /
  //            link.throughput_bps
  auto minmax_latency = fsec::max();
  std::valarray<float> amin(Nf);
  std::valarray<float> a_ratio(Nf);
  // auto afit = flows.begin();
  for (size_t i = 0; i < Nf; ++i) {
    FlowID const &fid = reorderingVector[i].first;
    FlowInfo const &af = reorderingVector[i].second.second;
    // What is the minimum amount of time needed to send this flow over this link? -> The smaller the better (I don't want to spend a lot of time sending ridiculously big flows)
    // Link-Flow mapping is done through _defaultLinkInfo in dll.cc
    amin[i] = (af.link_info.frame_overhead +
               fsec(af.min_segments_per_frame *
                    (af.link_info.segment_bit_overhead + af.bits_per_segment) /
                    af.link_info.throughput_bps))
                  .count();
    // G_j + (S + M_j)/x_j
    // -------------------
    // M_j/x_j
    // A ratio of the smallest amount of time taken to send this flow (inclusive of overheads) to the smallest amount of time taken to send just the payload for this flow OVER THE GIVEN LINK.
    // This should be as close to 1 as possible.
    a_ratio[i] =
        (af.link_info.frame_overhead +
         fsec(af.min_segments_per_frame *
              (af.link_info.segment_bit_overhead + af.bits_per_segment) /
              af.link_info.throughput_bps)) /
        fsec(af.min_segments_per_frame *
             (af.bits_per_segment - af.segment_payload_bit_overhead) /
             af.link_info.throughput_bps);
    af.im.visit<void>(
        [&](auto const &streampt) {
          // R_j / (y_j * beta_j)
          // Given the quality of the link (beta and link_goodput), how achievable is this STREAM flow?
          // Remember, we also take the overhead involvement into consideration (by multiplying {G_j + (S + M_j)/x_j} with {R_j / (y_j * beta_j)}).
          // This should be <= 1, the lower the better, implying we can achieve the min_throughput requirement for this STREAM flow given the quality of the link.
          a_ratio[i] *= streampt.min_throughput_bps /
                        (af.link_info.goodput_bps * af.beta);
          // The max_latency requirement for this goal
          auto const flow_max_latency = af.alpha * streampt.max_latency;
          // Finding the minimum of the "max_latency" requirement
          if (flow_max_latency < minmax_latency) {
            minmax_latency = flow_max_latency;
          }
        },
        [&](auto const &) {
          auto const &fffp = ffp.at(fid.flowUID());
          auto const trem = fffp.deadline - now;
          if (trem <= 0s) {
            a_ratio[i] = 0.0f;
          } else {
            // f_j / (t_j * β * z_j)
            // Given the number of bits remaining to be sent, the quality of the link (beta and link_goodput), and the time remaining until the deadline is reached, how achievable is this FILE_TRANSFER flow?
            // Remember, we also take the overhead involvement into consideration (by multiplying {G_j + (S + M_j)/x_j} with {f_j / (t_j * β * z_j)}).
            // This should be <= 1, the lower the better, implying we can transfer the remaining bits in this FILE_TRANSFER flow given the time remaining and the quality of the link.
            a_ratio[i] *=
                fffp.bits_remaining /
                (fsec(trem) * af.beta * af.link_info.goodput_arq_bps).count();
          }
        });
  }
  if (minmax_latency == fsec::max()) {
    // FIXME remove hardcoded 10s max
    minmax_latency = 10s;
  }
  auto const &Amax = minmax_latency;
  // The smallest total amount of time needed to send all the flows given to this node in this scheduling iteration.
  auto const Amin = amin.sum();
  // The upper bound - the smallest max_latency
  auto Au = Amax.count();
  // The lower bound - the smallest amount of time needed to send all these flows within this scheduling period
  auto Al = Amin;
  auto Ai = Au;
  std::valarray<float> a(Nf);
  // A default invalid initialization of QuantumSchedule with Amin as the lower_bound and Amax as the upper_bound
  QuantumSchedule qs{{}, false, 0s, fsec(Amin), Amax};
  // FIXME: Is 16 the right amount of leeway for flow-fitting?
  size_t const bisimax = 16;
  for (size_t bisi = 0; bisi < bisimax; ++bisi) {
    // Enforce two lower limits on quantums:
    // 1. Quantums must give sufficient rates.
    decltype(a) ai = Ai * a_ratio;
    // 2. Quantums must be larger than min frame size.
#if 1
    auto const r = ai < amin;
    ai[r] = amin[r];
#else
    for (size_t i = 0; i < Nf; ++i) {
      if (ai[i] < amin[i]) {
        ai[i] = amin[i];
      }
    }
#endif
    // Check sum of lower bounded quantums is under the current iteration's sum round time.
    auto const Asumi = ai.sum();
    // Can I fit all the flows within the current min max_latency? If yes, is the upper bound too high? If yes, lower it. If not, done.
    // Can I fit all the flows within the current min max_latency? If not, is it the very first iteration here? If yes, the sum round time is at upper bound, exit. If not, increase the sum round time a little and try again.
    if (Asumi <= Ai) {
      // If yes, the array Ai * a_ratio gives me the quantums for each flow in this node (tree root or any other subsequent child nodes)
      a = ai;
      qs.valid = true;
      // The period member in the QuantumSchedule struct is the sum of Ai * aratio
      qs.period = fsec(Asumi);
      if (Ai - Asumi > (float)5e-6) {
        // Sum of lower bounded quantums is under the current iteration's sum round time -> Tighten sum round time with the arithmetic mean update
        Au = Ai;
      } else {
        break;
      }
    } else {
      // Sum of lower bounded quantums exceeds the current iteration's sum round time -> Current iteration sum round time is too low -> Up the lower limit
      if (bisi == 0) {
        assert(Asumi > Amax.count());
        // On first iteration, sum round time is at upper bound. jk/su
        a = ai;
        qs.valid = false;
        qs.period = fsec(Asumi);
        break;
      }
      // A change in Al for arithmetic mean update
      Al = Ai;
    }
    assert(Au - Al > (float)1e-7);
    // Use the arithmetic mean of the updated lower or upper bounds to dictate the current iteration's sum round time
    Ai = (Al + Au) / 2.0f;
  }
  // afit = flows.begin();
  // SUCCESS: Valid schedule - Assigning quantums for each individual flow in this node
  // FIXME: Is reordering even necessary? Does it matter?
  for (size_t i = 0; i < Nf; ++i) {
    FlowID const &flow_id = reorderingVector[i].first;
    qs.quantums.emplace(flow_id,
                        chrono::duration_cast<chrono::nanoseconds>(fsec(a[i])));
  }
  // Update the upper bound value (if it was changed within the bisi loop) for duty-cycle evaluation during branching
  qs.periodub = fsec(Ai);
  return qs;
}

/// A custom sorting routine for the value/Hz reordering
bool customSort(const std::pair< FlowID, std::pair< float, FlowInfo > > &a, const std::pair< FlowID, std::pair< float, FlowInfo > > &b) {
  return a.second.first > b.second.first;
}

/*
   * Create a graph of nodes. Each node is a set of flows. Child nodes are the
   * subset of their parents with one flow removed. The removed flow is either
   * a max quantum or min max_latency flow in the parent.
   *
   * Starting with the all-flows node, perform a breadth-first search of the
   * graph until a valid schedule is found for a node.
 */
QuantumSchedule scheduleMaxFlows(FlowInfoMap const &allfim,
                                 std::map<FlowUID, FileFlowProgress> const &ffp,
                                 std::chrono::system_clock::time_point now,
                                 MaxFlowsSearch search, bool respectPriority) {
  // An initial validation check
  if (allfim.empty()) {
    return QuantumSchedule{{}, true, 0s, 0s, 0s};
  }
  struct FlowSetNode {
    FlowInfoMap fim;
    inline size_t level() const { return fim.size(); }
  };
  // Frontier of nodes to visit.
  std::queue<FlowSetNode> frontier;
  // Visited node set.
  std::set<FlowInfoMap> visited;
  // A vector of pairs for the reordering operation
  std::vector< std::pair< FlowID, std::pair< float, FlowInfo > > > reorderingVector;
  // Create and add root node with all flows.
  frontier.push(FlowSetNode{allfim});
  while (!frontier.empty()) {
    FlowSetNode subroot = std::move(frontier.front());
    frontier.pop();
    // Already visited?
    if (visited.count(subroot.fim) > 0) {
      continue;
    }
    FlowInfoMap &fim = subroot.fim;
    // An intermediate check - a safety measure
    if (fim.empty()) {
      continue;
    }
    // Clear the reordering vector for this subroot
    reorderingVector.clear();
    // Reordering of flows according to the value/Hz evaluation
    std::map<FlowID, FlowInfo const *>::iterator it;
    for (it = fim.begin(); it != fim.end(); ++it) {
      FlowID const &flowID = it->first;
      FlowInfo const &individualFlowInfo = *it->second;
      // Use priority here instead of point_value - required for differentiating AllNodes flows with regular flows (it's just scaling)...
      reorderingVector.push_back(std::make_pair(flowID, std::make_pair((individualFlowInfo.priority / individualFlowInfo.link_info.channel_bandwidth), individualFlowInfo)));
    }
    // Reorder the flows of this subroot according to the evaluated 1/(value/Hz) metric 
    sort(reorderingVector.begin(), reorderingVector.end(), customSort);
    // The signature of this routine has been changed to accomodate the value/Hz reordering
    QuantumSchedule const qs = scheduleFlows(reorderingVector, ffp, now);
    // Return valid QuantumSchedule structs
    if (qs.valid) {
      return qs;
    }
    // If I didn't get a valid schedule from the previous node, branch.
    // Branch in two directions:
    // 1) Remove lowest value/Hz/duty-cycle flow (break tie: remove min max_latency) because this flow is not worth scheduling
    // 2) Remove lowest latency flow (break tie: remove min value/Hz/duty-cycle) because this is probably unattainable
    // 3) A transient branch: Remove max quantum flow (break tie: remove min max_latency) because this is not worth scheduling
    // Breaking ties is more expensive, so add a switch to disable.
    bool constexpr breaktie = true;
    FlowID minlatencyflowuid;
    if ((uint64_t)search & (uint64_t)MaxFlowsSearch::RemoveMinMaxLatency) {
      // Here, a priority check with ties broken using max_latencies (with ties broken using quantums)
      if (breaktie) {
        minlatencyflowuid =
            std::min_element(
                fim.begin(), fim.end(),
                [&qs, respectPriority](FlowInfoMap::value_type const &lhsfip,
                                       FlowInfoMap::value_type const &rhsfip) {
                  FlowInfo const &lhsfi = *lhsfip.second;
                  FlowInfo const &rhsfi = *rhsfip.second;
                  if (respectPriority) {
                    // Low priority flows are "less" than high priority.
                    if (lhsfi.priority < rhsfi.priority) {
                      return true;
                    } else if (lhsfi.priority > rhsfi.priority) {
                      return false;
                    }
                    // equal priorities -> compare max_latencies
                  }
                  // Get lhs and rhs latency.
                  auto const lhsl = lhsfi.im.visit<fsec>(
                      [&](auto const &streampt) {
                        return lhsfi.alpha * streampt.max_latency;
                      },
                      [](IndividualMandate::FilePT const &) {
                        return fsec::max();
                      });
                  auto const rhsl = rhsfi.im.visit<fsec>(
                      [&](auto const &streampt) {
                        return rhsfi.alpha * streampt.max_latency;
                      },
                      [](IndividualMandate::FilePT const &) {
                        return fsec::max();
                      });
                  // Compare latencies and break ties using quantums
                  if (lhsl < rhsl) {
                    return true;
                  } else if (lhsl > rhsl) {
                    return false;
                  } else {
                    if (qs.quantums.at(lhsfip.first) >
                        qs.quantums.at(rhsfip.first)) {
                      return true;
                    } else {
                      return false;
                    }
                  }
                })
                // Return FlowID of the flow with the min max_latency with tie-break enabled
                ->first;
      } else {
        minlatencyflowuid =
            std::min_element(
                fim.begin(), fim.end(),
                [respectPriority](FlowInfoMap::value_type const &lhsfip,
                                  FlowInfoMap::value_type const &rhsfip) {
                  FlowInfo const &lhsfi = *lhsfip.second;
                  FlowInfo const &rhsfi = *rhsfip.second;
                  if (respectPriority) {
                    // Low priority flows are "less" than high priority.
                    if (lhsfi.priority < rhsfi.priority) {
                      return true;
                    } else if (lhsfi.priority > rhsfi.priority) {
                      return false;
                    }
                    // equal priorities -> compare max_latencies
                  }
                  auto const lhsl = lhsfi.im.visit<fsec>(
                      [&](auto const &streampt) {
                        return lhsfi.alpha * streampt.max_latency;
                      },
                      [](IndividualMandate::FilePT const &) {
                        return fsec::max();
                      });
                  auto const rhsl = rhsfi.im.visit<fsec>(
                      [&](auto const &streampt) {
                        return rhsfi.alpha * streampt.max_latency;
                      },
                      [](IndividualMandate::FilePT const &) {
                        return fsec::max();
                      });
                  // No tie break
                  return lhsl < rhsl;
                })
                // Return FlowID of the flow with the min max_latency with tie-break disabled
                ->first;
      }
      FlowSetNode child = subroot;
      child.fim.erase(minlatencyflowuid);
      assert(visited.count(child.fim) == 0);
      frontier.emplace(std::move(child));
    } // One branch -> Create a child node with min max_latency flow removed
    FlowID maxquantflowuid;
    if ((uint64_t)search & (uint64_t)MaxFlowsSearch::RemoveMaxQuantum) {
      // Here, priority check with ties broken using quantums (with ties broken using max_latencies)
      if (breaktie) {
        // TODO more efficient to iterate over qs or fim?
        // FIXME Iterate over the map and use the key to reference into the returned QuantumSchedule
        maxquantflowuid =
            std::max_element(
                qs.quantums.begin(), qs.quantums.end(),
                [&fim, respectPriority](
                    std::pair<FlowID, chrono::nanoseconds> const &lhs,
                    std::pair<FlowID, chrono::nanoseconds> const &rhs) {
                  if (respectPriority) {
                    // max_element is O(N), but each < comparison is made O(log n) by these map lookups... not ideal. FIXME
                    auto const &lhsfi = *fim.at(lhs.first);
                    auto const &rhsfi = *fim.at(rhs.first);
                    // High priority flows are "less" than low priority.
                    if (lhsfi.priority > rhsfi.priority) {
                      return true;
                    } else if (lhsfi.priority < rhsfi.priority) {
                      return false;
                    }
                    // equal priorities -> compare quantums
                  }
                  // TODO fuzzy compare/tie determination?
                  if (lhs.second < rhs.second) {
                    return true;
                  } else if (lhs.second > rhs.second) {
                    return false;
                  } else {
                    // Hopefully ties are infrequent so these map::at() lookups are rare.
                    auto const &lhsfi = *fim.at(lhs.first);
                    auto const &rhsfi = *fim.at(rhs.first);
                    auto const lhsl = lhsfi.im.visit<fsec>(
                        [&](auto const &streampt) {
                          return lhsfi.alpha * streampt.max_latency;
                        },
                        [](IndividualMandate::FilePT const &) {
                          return fsec::max();
                        });
                    auto const rhsl = rhsfi.im.visit<fsec>(
                        [&](auto const &streampt) {
                          return rhsfi.alpha * streampt.max_latency;
                        },
                        [](IndividualMandate::FilePT const &) {
                          return fsec::max();
                        });
                    return rhsl < lhsl;
                  }
                // Return FlowID of the flow with the max quantum with tie-break enabled
                })->first;
      } else {
        maxquantflowuid =
            std::max_element(
                qs.quantums.begin(), qs.quantums.end(),
                [&fim, respectPriority](auto const &lhs, auto const &rhs) {
                  if (respectPriority) {
                    // max_element is O(N), but each < comparison is made
                    // O(log n) by these map lookups... not ideal. FIXME
                    auto const &lhsfi = *fim.at(lhs.first);
                    auto const &rhsfi = *fim.at(rhs.first);
                    // High priority flows are "less" than low priority.
                    if (lhsfi.priority > rhsfi.priority) {
                      return true;
                    } else if (lhsfi.priority < rhsfi.priority) {
                      return false;
                    }
                    // equal priorities -> compare quantums
                  }
                  return lhs.second < rhs.second;
                // Return FlowID of the flow with the max quantum with tie-break disabled
                })->first;
      }
      if (((uint64_t)search & (uint64_t)MaxFlowsSearch::RemoveMinMaxLatency) && (minlatencyflowuid == maxquantflowuid)) {
        // Already removed
        continue;
      }
      // If not already removed, remove it now.
      FlowSetNode child = subroot;
      child.fim.erase(maxquantflowuid);
      assert(visited.count(child.fim) == 0);
      frontier.emplace(std::move(child));
    }//Another branch (transient): Create a child node with the max quantum flow removed
    if ((uint64_t)search & (uint64_t)MaxFlowsSearch::RemoveMinValue) {
      auto upper_bound = qs.periodub.count();
      FlowID minvalueflowuid;
      // Here, value/Hz/Duty-Cycle check with ties broken using max_latencies
      if (breaktie) {
        minvalueflowuid = std::min_element(qs.quantums.begin(), qs.quantums.end(), [&upper_bound, &fim](std::pair<FlowID, chrono::nanoseconds> const &lhs, std::pair<FlowID, chrono::nanoseconds> const &rhs) {
          auto const &lhsfi = *fim.at(lhs.first);
          auto const &rhsfi = *fim.at(rhs.first);
          if ((lhsfi.priority / (lhsfi.link_info.channel_bandwidth * (lhs.second.count() / upper_bound))) < (rhsfi.priority / (rhsfi.link_info.channel_bandwidth * (rhs.second.count() / upper_bound)))) {
            return true;
          } else if ((lhsfi.priority / (lhsfi.link_info.channel_bandwidth * (lhs.second.count() / upper_bound))) > (rhsfi.priority / (rhsfi.link_info.channel_bandwidth * (rhs.second.count() / upper_bound)))) {
            return false;
          } else {
            auto const lhsl = lhsfi.im.visit<fsec>(
                [&](auto const &streampt) {
                  return lhsfi.alpha * streampt.max_latency;
                },
                [](IndividualMandate::FilePT const &) {
                  return fsec::max();
                }
            );
            auto const rhsl = rhsfi.im.visit<fsec>(
                [&](auto const &streampt) {
                  return rhsfi.alpha * streampt.max_latency;
                },
                [](IndividualMandate::FilePT const &) {
                  return fsec::max();
                }
            );
            return rhsl < lhsl;
          }
        // Return FlowID of the flow with the least value/Hz/duty-cycle with tie-break enabled
        })->first;
      } else {
        minvalueflowuid = std::min_element(qs.quantums.begin(), qs.quantums.end(), [&upper_bound, &fim](std::pair<FlowID, chrono::nanoseconds> const &lhs, std::pair<FlowID, chrono::nanoseconds> const &rhs) {
          auto const &lhsfi = *fim.at(lhs.first);
          auto const &rhsfi = *fim.at(rhs.first);
          if ((lhsfi.priority / (lhsfi.link_info.channel_bandwidth * (lhs.second.count() / upper_bound))) < (rhsfi.priority / (rhsfi.link_info.channel_bandwidth * (rhs.second.count() / upper_bound)))) {
            return true;
          }
          return false;
        // Return FlowID of the flow with the least value/Hz/duty-cycle with tie-break disabled
        })->first;
      }
      if ((((uint64_t)search & (uint64_t)MaxFlowsSearch::RemoveMinMaxLatency) && (minlatencyflowuid == minvalueflowuid)) || (((uint64_t)search & (uint64_t)MaxFlowsSearch::RemoveMaxQuantum) && (maxquantflowuid == minvalueflowuid))) {
          // Already removed
          continue;
      }
      // If not already removed, remove it now.
      FlowSetNode child = subroot;
      child.fim.erase(minvalueflowuid);
      assert(visited.count(child.fim) == 0);
      frontier.emplace(std::move(child)); 
    } // Another branch -> Create a child node with min value/Hz/duty-cycle removed
    // Done visiting this node
    visited.emplace(std::move(fim));
  }
  return QuantumSchedule{{}, true, 0s, 0s, 0s};
}

bool operator==(QuantumSchedule const &lhs, QuantumSchedule const &rhs) {
  return lhs.valid == rhs.valid && lhs.period == rhs.period &&
         lhs.periodlb == rhs.periodlb && lhs.periodub == rhs.periodub &&
         lhs.quantums == rhs.quantums;
}
bool operator!=(QuantumSchedule const &lhs, QuantumSchedule const &rhs) {
  return lhs.valid != rhs.valid || lhs.period != rhs.period ||
         lhs.periodlb != rhs.periodlb || lhs.periodub != rhs.periodub ||
         lhs.quantums != rhs.quantums;
}

/// The following is a sample IM representation in JSON format
/* Each goal is a JSON dict:
  {"goal_type": "Traffic",
  "flow_uid": 15037,
  "point_value": 15,
  "requirements": {
    "min_throughput_bps": 40560.0,
    "max_latency_s": 0.37,
    "max_packet_drop_rate": 0.1}
  }

  {"goal_type": "Traffic",
  "flow_uid": 15042,
  "point_value": 1
  "requirements": {
    "file_size_bytes": 655360,
    "file_transfer_deadline_s": 10.0,
    "max_packet_drop_rate": 0.0}
  }
*/
/// Added a point value member to the bamradio::IndividualMandate struct for prioritized flow scheduling
std::map<FlowUID, IndividualMandate>
IndividualMandate::fromJSON(nlohmann::json const &mandates) {
  std::map<FlowUID, IndividualMandate> flows;
  for (auto const &goal : mandates) {
    auto const flowUIDRef = goal["flow_uid"];
    auto const pointValue = goal["point_value"];
    // FlowUID compliance check
    if (!flowUIDRef.is_number_unsigned()) {
      panic("IM flow_uid must be unsigned number");
    }
    // FlowPointValue compliance check
    if (!pointValue.is_number_unsigned()) {
      panic("The point value of a flow must be an unsigned number");
    }
    auto const flowUID = flowUIDRef.get<FlowUID>();
    auto const &requirements = goal["requirements"];
    auto const latencyIt = requirements.find("max_latency_s");
    auto const throughputIt = requirements.find("min_throughput_bps");
    auto const sizeIt = requirements.find("file_size_bytes");
    auto const durationIt = requirements.find("file_transfer_deadline_s");
    if (latencyIt != requirements.end() && throughputIt != requirements.end()) {
      if (!latencyIt->is_number_float() || !throughputIt->is_number_float()) {
        panic("IM latency and throughput must be floats");
      }
      // A stream flow (with the newly added pointValue member)
      flows.emplace(flowUID, IndividualMandate{pointValue, IndividualMandate::StreamPT{
                                 throughputIt->get<float>(),
                                 fsec(latencyIt->get<float>())}});
    } else if (durationIt != requirements.end()) {
      ssize_t size_bytes = -1;
      if (sizeIt != requirements.end() && sizeIt->is_number_unsigned()) {
        size_bytes = sizeIt->get<size_t>();
      }
      if (!durationIt->is_number_float()) {
        panic("file_transfer_deadline_s not a float");
      }
      // A file flow (with the newly added pointValue member)
      flows.emplace(flowUID, IndividualMandate{pointValue, IndividualMandate::FilePT{
                                 size_bytes, fsec(durationIt->get<float>())}});
    } else {
      panic((format("Unrecognized requirements: %s") % requirements.dump())
                .str());
    }
  }
  return flows;
}

} // namespace bamradio

