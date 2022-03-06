/* -*- c++ -*- */
// Copyright (c) 2017 Tomohiro Arakawa <tarakawa@purdue.edu>.

#ifndef CONTROLLER_SRC_NETWORKMAP_H_
#define CONTROLLER_SRC_NETWORKMAP_H_

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <vector>
#include <map>
#include <memory>
#include <tuple>

namespace bamradio {

class NetworkMap {
public:
  NetworkMap();
  virtual ~NetworkMap() {
  }
  ;
  typedef std::shared_ptr<NetworkMap> sptr;

  /* Boost Graph Library */
  typedef uint8_t SRNID;
  typedef boost::property<boost::edge_weight_t, double> EdgeProperty;
  typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS,
      SRNID, EdgeProperty> Graph;
  typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
  typedef std::pair<uint8_t, uint8_t> Edge;

  /**
   * @brief Set link weight
   * @param srn_from Source node SRN ID
   * @param srn_to Destination node SRN ID
   * @param cost Weight of the link
   */
  void setLink(uint8_t srn_from, uint8_t srn_to, double cost);

  /**
   * @brief Remove link
   * @param srn_from Source node SRN ID
   * @param srn_to Destination node SRN ID
   * @return true when the link is successfully removed
   */
  bool removeLink(uint8_t srn_from, uint8_t srn_to);

  /**
   * @brief Check if the specified link exists
   * @param srn_from Source node SRN ID
   * @param srn_to Destination node SRN ID
   * @return True: Link up, False: Link down
   */
  bool isLinkUp(uint8_t srn_from, uint8_t srn_to);

  /**
   * @brief Get network graph
   * @return Network graph
   */
  Graph getGraph();

  /**
   * @brief Get all SRN IDs
   * @return SRN IDs
   */
  std::vector<uint8_t> getAllSrnIds();

  /**
   * Get vertex_descriptor from SRN ID
   * @param SRN ID
   * @return vertex_descriptor. Returns null_vertex if the node does not exist
   */
  Vertex getNode(uint8_t srn);

  // serialization
  std::vector<char> serializeToVector() const;
  // deserialize from vector
  NetworkMap(std::vector<char> const &bytes);

private:
  Graph d_g;
};

} /* namespace bamradio */

#endif /* CONTROLLER_SRC_NETWORKMAP_H_ */
