#include "../src/arqbuffer.h"
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

int main(int argc, char const *argv[]) {
  using namespace bamradio;
  ARQBuffer arqbuf(100);
  arqbuf.add(122, std::make_shared<std::vector<uint8_t>>(),
             std::chrono::seconds(2));
  arqbuf.add(123, std::make_shared<std::vector<uint8_t>>(),
             std::chrono::seconds(10));
  arqbuf.add(125, std::make_shared<std::vector<uint8_t>>(),
             std::chrono::seconds(8));
  arqbuf.add(129, std::make_shared<std::vector<uint8_t>>(),
             std::chrono::seconds(8));
  arqbuf.printDebugMsg();
  auto vec = arqbuf.getMissingSeqNums();
  for (auto elem : vec) std::cout << elem << std::endl;
}
