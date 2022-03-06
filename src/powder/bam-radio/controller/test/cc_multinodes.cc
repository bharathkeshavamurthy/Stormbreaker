/*
 * Multi-node CCData test
 * Copyright (c) 2018 Tomohiro Arakawa <tarakawa@purdue.edu>
 */

#include "../src/cc_data.h"

#include <boost/asio.hpp>
#include <vector>

int main(int argc, char const *argv[]) {
  using namespace bamradio;

  // Number of nodes to simulate
  size_t const n_nodes = 10;

  assert(n_nodes > 1);

  // Generate CCData (Node ID 0 to n_nodes - 1)
  std::vector<controlchannel::CCData::sptr> cc_data_vec;
  for (size_t i = 0; i < n_nodes; ++i) {
    // Node 0 is the gateway
    bool const is_gateway = (i == 0);
    // Instantiate CCData
    auto cc_data = std::make_shared<controlchannel::CCData>(i, 1, is_gateway);
    // Push
    cc_data_vec.push_back(cc_data);
  }

  // Modify CCData contents
  // for instance...
  cc_data_vec[0]->setLocation(100, 50, 5);
  cc_data_vec[1]->setLocation(50, 10, 0);

  // Send all non-gateway CCData info to the gateway
  for (size_t i = 1; i < n_nodes; ++i) {
    auto const serialized_data = cc_data_vec[i]->serialize();
    auto cbuf =
        boost::asio::buffer(serialized_data->data(), serialized_data->size());
    cc_data_vec[0]->deserialize(cbuf, false);
  }

  // For covenience...
  auto ccdata_gw = cc_data_vec[0];

  // Now it's ready to use the gateway's CCData
  auto srnids = ccdata_gw->getAllSRNIDs();
  for (auto &v : srnids) {
    std::cout << "Node ID: " << (int)v << std::endl;
  }
}
