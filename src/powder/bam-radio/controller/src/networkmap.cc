/* -*- c++ -*- */
/* Network map
 * Copyright (c) 2017 Tomohiro Arakawa <tarakawa@purdue.edu>.
 */

#include "networkmap.h"
#include <algorithm>
#include <iostream>

#include <boost/graph/adj_list_serialize.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/device/back_inserter.hpp>

using namespace boost;

namespace bamradio {

NetworkMap::NetworkMap() {
}

void NetworkMap::setLink(uint8_t srn_from, uint8_t srn_to, double cost) {
  // src node
  Vertex v_from = getNode(srn_from);
  if (v_from == d_g.null_vertex())
    v_from = add_vertex(srn_from, d_g);
  // dst node
  Vertex v_to = getNode(srn_to);
  if (v_to == d_g.null_vertex())
    v_to = add_vertex(srn_to, d_g);

  // check if link exists
  if (isLinkUp(srn_from, srn_to)) {
    // update cost if link exists
    auto ed = edge(v_from, v_to, d_g).first;
    put(edge_weight_t(), d_g, ed, cost);
  } else
    // add new link if no edge exists
    add_edge(v_from, v_to, cost, d_g);
}

bool NetworkMap::isLinkUp(uint8_t srn_from, uint8_t srn_to) {
  Vertex v_from = getNode(srn_from);
  if (v_from == d_g.null_vertex())
    return false; // no src node
  Vertex v_to = getNode(srn_to);
  if (v_to == d_g.null_vertex())
    return false; // no dst node
  // return true if link exists
  return edge(v_from, v_to, d_g).second;
}

NetworkMap::Graph NetworkMap::getGraph() {
  return d_g;
}

std::vector<uint8_t> NetworkMap::getAllSrnIds() {
  typedef graph_traits<Graph>::vertex_iterator vertex_iter;
  std::vector<uint8_t> nodes;
  std::pair<vertex_iter, vertex_iter> vp;
  for (vp = vertices(d_g); vp.first != vp.second; ++vp.first)
    nodes.push_back(d_g[*vp.first]);
  return nodes;
}

NetworkMap::Vertex NetworkMap::getNode(uint8_t srn) {
  typedef graph_traits<Graph>::vertex_iterator vertex_iter;
  std::pair<vertex_iter, vertex_iter> vp;
  for (vp = vertices(d_g); vp.first != vp.second; ++vp.first)
    if (d_g[*vp.first] == srn)
      return *vp.first;
  return d_g.null_vertex();
}

bool NetworkMap::removeLink(uint8_t srn_from, uint8_t srn_to) {
  // return false if the specified link does not exist
  if (!isLinkUp(srn_from, srn_to))
    return false;

  // remove edge
  Vertex v_from = getNode(srn_from);
  Vertex v_to = getNode(srn_to);
  remove_edge(v_from, v_to, d_g);
  return true;
}

// https://stackoverflow.com/a/5604782
//
// n.b. I think this whole thing would be nicer if we used std::vector<uint8_t>
// instead of vector<char> for this, but the back_insert_device does not work
// for uint8_t. seems like boost iostream requires this to be char. oh well...
std::vector<char> NetworkMap::serializeToVector() const {
  std::vector<char> out;
  iostreams::back_insert_device<decltype(out)> inserter(out);
  iostreams::stream<decltype(inserter)> s(inserter);
  archive::binary_oarchive o(s);
  o << d_g;
  s.flush();
  return out;
};

NetworkMap::NetworkMap(std::vector<char> const & bytes) {
  iostreams::array_source src(bytes.data(), bytes.size());
  iostreams::stream<decltype(src)> s(src);
  archive::binary_iarchive i(s);
  i >> d_g;
}

} /* namespace bamradio */
