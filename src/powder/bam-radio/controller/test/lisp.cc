// lisp unit tests
//
// Copyright (c) 2019 Dennis Ogbe

#define BOOST_TEST_MODULE lisp
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <boost/format.hpp>
#include <iostream>
#include <random>
#include <sstream>

#include "lisp.h"

#define BE_VERBOSE 1

//
// Test Initialization
//

using namespace lisp;

// initialize and shut down the lisp environment

struct infra {
  infra() : env({}) {}
  ~infra() {}

  lisp::Environment env;
};

BOOST_GLOBAL_FIXTURE(infra);

//
// Test Helpers
//

// test equality, defaults to equalp
bool lispEquality(cl_object a, cl_object b,
                  std::string const &type = "equalp") {
  auto ret = Funcall(Symbol(type), a, b);
  return ret == lisp::t;
}

// basic equality test
template <typename T>
inline void testToLisp(T const x, std::string const &exp,
                       std::string const &lispeq = "equalp") {
  auto const converted = toLisp(x);
  auto const expected = ReadEval(exp);
#ifdef BE_VERBOSE
  std::cout << "converted: ";
  Print(converted);
  std::cout << " expected: ";
  Print(expected);
  std::cout << "(type-of converted) => ";
  Print(Funcall(Symbol("type-of"), converted));
  std::cout << "(type-of expected)  => ";
  Print(Funcall(Symbol("type-of"), expected));
  std::cout << std::endl;
#endif
  BOOST_REQUIRE(lispEquality(converted, expected));
}

//
// Tests
//

BOOST_AUTO_TEST_CASE(boolean) {
  testToLisp(false, "NIL");
  testToLisp(true, "T");
}

BOOST_AUTO_TEST_CASE(complex) {
  testToLisp(std::complex<float>(1.0, 2.0), "#C(1.0 2.0)");
  testToLisp(std::complex<double>(1000.5, 5000.45), "#C(1000.5d0 5000.45d0)");
}

BOOST_AUTO_TEST_CASE(string) {
  const char *str = "this is a string";
  const char *expected = "\"this is a string\"";
  testToLisp(str, expected);
  testToLisp(std::string(str), expected);
  testToLisp("this is a string", expected);
}

BOOST_AUTO_TEST_CASE(float_vector) {
  std::vector<float> fv = {1.1, 1.2, 1.3, 1.4, 4000.4444};
  testToLisp(fv, "#(1.1 1.2 1.3 1.4 4000.4444)");
}

BOOST_AUTO_TEST_CASE(int_vector) {
  std::vector<int64_t> iv = {1, 2, 3, 4, 5000};
  testToLisp(iv, "#(1 2 3 4 5000)");
}

BOOST_AUTO_TEST_CASE(deque) {
  std::deque<int64_t> deck;
  std::stringstream expected;
  std::mt19937 rng(33);
  std::uniform_int_distribution<int64_t> dist(-100, 100);
  expected << "#(";
  for (size_t i = 0; i < 10; ++i) {
    auto const num = dist(rng);
    deck.push_back(num);
    expected << num << " ";
  }
  expected << ")";
  testToLisp(deck, expected.str());
}

BOOST_AUTO_TEST_CASE(bit_vector) {
  std::vector<bool> bv = {0, 1, 0, 1, 0, 1, 0, 1, 1, 1};
  testToLisp(bv, "#*0101010111", "equal");
}

BOOST_AUTO_TEST_CASE(bitset) {
  std::bitset<10> bs("1110101010"); // beware of endianness!
  testToLisp(bs, "#*0101010111", "equal");
}

BOOST_AUTO_TEST_CASE(array) {
  std::array<double, 5> arr{{1.1, 1.2, 1.3, 1.4, 4000.4444}};
  testToLisp(arr, "#(1.1d0 1.2d0 1.3d0 1.4d0 4000.4444d0)");
}

BOOST_AUTO_TEST_CASE(list) {
  std::list<std::string> list{"the", "frogurt", "is", "also", "cursed"};
  testToLisp(list, "'(\"the\" \"frogurt\" \"is\" \"also\" \"cursed\")");
}

BOOST_AUTO_TEST_CASE(hash_table) {
#ifdef BE_VERBOSE
  // debug output for hash tables
  Funcall(Symbol("make-package"), toLisp(util::upcase("test-package-1")));
  ReadEval("(defun TEST-PACKAGE-1::PRINT-HASH-TABLE (ht &optional stream)"
           "  (format (or stream t) \"#HASH{~{~{(~a : ~a)~}~^ ~}}\""
           "          (loop for key being the hash-keys of ht"
           "             using (hash-value value)"
           "             collect (list key value))))");
#endif

  using std::string;
  std::map<string, int> m1;
  m1["foo"] = 70;
  m1["bar"] = 71;
  m1["baz"] = 72;
  auto expected1 = "(let ((ht (make-hash-table :test 'equal)))"
                   "  (setf (gethash \"foo\" ht) 70)"
                   "  (setf (gethash \"bar\" ht) 71)"
                   "  (setf (gethash \"baz\" ht) 72)"
                   " ht)";
#ifdef BE_VERBOSE
  std::cout << "converted: " << std::flush;
  Funcall(Symbol("print-hash-table", "test-package-1"), toLisp(m1));
  std::cout << std::endl;
  std::cout << " expected: " << std::flush;
  Funcall(Symbol("print-hash-table", "test-package-1"), ReadEval(expected1));
#endif
  testToLisp(m1, expected1);

  std::map<int, string> m2;
  m2[70] = "foo";
  m2[71] = "bar";
  m2[72] = "baz";
  auto expected2 = "(let ((ht (make-hash-table)))"
                   "  (setf (gethash 70 ht) \"foo\")"
                   "  (setf (gethash 71 ht) \"bar\")"
                   "  (setf (gethash 72 ht) \"baz\")"
                   " ht)";
#ifdef BE_VERBOSE
  std::cout << "converted: " << std::flush;
  Funcall(Symbol("print-hash-table", "test-package-1"), toLisp(m2));
  std::cout << std::endl;
  std::cout << " expected: " << std::flush;
  Funcall(Symbol("print-hash-table", "test-package-1"), ReadEval(expected2));
#endif
  testToLisp(m2, expected2);
}

// lispy method
class WITH_LISPY_METHODS(lmtest_class) {
public:
  LISPY_METHOD(setInteger, int64_t, fromInt);
  int64_t number() const { return number_; };

private:
  int64_t number_;
};

void lmtest_class::setInteger(int64_t data) { number_ = data; }

BOOST_AUTO_TEST_CASE(lispy_method) {
  // make the test class. we want to set the number_ member to the value of
  // "expected" from lisp.
  lmtest_class lmtest;
  int64_t const expected = 42;

  // make a new package and add the lispy setter method to it
  Funcall(Symbol("make-package"), toLisp(util::upcase("test-package-2")));
  auto setInteger = Symbol("set-integer", "test-package-2");
  addFunction(setInteger, lmtest_class::setInteger_LISPMETHOD);

  // call the lispy method and make sure that the C++ object was modified
  BOOST_REQUIRE(expected != lmtest.number());
  Funcall(setInteger, CPointer(&lmtest), toLisp(expected));
  BOOST_REQUIRE_EQUAL(expected, lmtest.number());
}

// fromLisp
BOOST_AUTO_TEST_CASE(from_string) {
  auto const str = ReadEval("\"this is a string\"");
  auto const expected = "this is a string";
  BOOST_REQUIRE(fromString(str) == expected);
}

// map and filter

BOOST_AUTO_TEST_CASE(map_list) {
  auto const list = ReadEval("'(40 50 60 70)");
  std::vector<int64_t> const expected = {41, 51, 61, 71};
  auto const out =
      map(list, [](auto const &elem) { return 1 + fromInt(elem); });
  BOOST_REQUIRE(out == expected);
}

BOOST_AUTO_TEST_CASE(map_vector) {
  auto const list = ReadEval("#(40 50 60 70)");
  std::vector<int64_t> const expected = {41, 51, 61, 71};
  auto const out =
      map(list, [](auto const &elem) { return 1 + fromInt(elem); });
  BOOST_REQUIRE(out == expected);
}

BOOST_AUTO_TEST_CASE(filter_list) {
  auto const list = ReadEval("'(4 5 6 7)");
  std::vector<int64_t> const expected = {4, 6};
  auto const out = filter(
      list, fromInt, [](auto const &num) { return fromInt(num) % 2 == 0; });
  BOOST_REQUIRE(out == expected);
}

BOOST_AUTO_TEST_CASE(filter_vector) {
  auto const list = ReadEval("#(4 5 6 7)");
  std::vector<int64_t> const expected = {4, 6};
  auto const out = filter(
      list, fromInt, [](auto const &num) { return fromInt(num) % 2 == 0; });
  BOOST_REQUIRE(out == expected);
}
