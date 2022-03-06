// -*- c++ -*-
//
// ECL/C++ Interface
//
// Copyright (c) 2019 Dennis Ogbe
//
// These are some utilities and hacks that are useful when embedding ECL in a
// C++ program.
//
// <2019-06-16 Sun> This file is known to work with ECL 16.1.3

#ifndef b00d9d87c4e5e57833e963e3b
#define b00d9d87c4e5e57833e963e3b

#include <algorithm>
#include <bitset>
#include <cctype>
#include <complex>
#include <cstdint>
#include <deque>
#include <list>
#include <map>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#if __has_include(<boost/asio.hpp>)
#include <boost/asio.hpp>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#endif

#include <ecl/ecl.h>

namespace lisp {

//
// Handling Lisp data in C++
//

// truth and falsehood
cl_object const t = ECL_T;
cl_object const nil = ECL_NIL;

// print representation of object to stdout
inline void Print(cl_object obj) { ecl_print(obj, ECL_T); }

// make a lisp list. nicer in c++.
template <typename... Args> inline cl_object List(Args... objs) {
  constexpr auto n = sizeof...(Args);
  return cl_list(n, objs...);
}

// funcall convenience wrapper
template <typename Fun, typename... Args>
inline cl_object Funcall(Fun fun, Args... args) {
  constexpr auto n = sizeof...(Args) + 1;
  return cl_funcall(n, fun, args...);
}

// evaluate a form
inline cl_object Eval(cl_object form) {
  return si_safe_eval(3, form, ECL_NIL, ECL_NIL);
}

// read and evaluate a string
inline cl_object ReadEval(std::string const &str) {
  return Eval(c_string_to_object(str.c_str()));
}

// get a symbol's value
inline cl_object Value(cl_object sym) { return cl_symbol_value(sym); }

// put a C function pointer in the function cell of symbol sym. the signature is
// cl_object function(cl_narg narg, ...) (see the lispy_method macro for usage)
inline void addFunction(cl_object sym, cl_objectfn c_function) {
  ecl_def_c_function_va(sym, c_function);
}

// upcase a string. a random helper function, useful below
namespace util {
inline std::string upcase(std::string const &s) {
  std::string o;
  o.reserve(s.length());
  std::transform(s.begin(), s.end(), std::back_inserter(o),
                 [](auto c) { return std::toupper(c); });
  return o;
}
} // namespace util

// find or intern symbol "sym" in "package"
inline cl_object Symbol(std::string const &sym,
                        std::string const &package = "CL") {
  return ecl_make_symbol(util::upcase(sym).c_str(),
                         util::upcase(package).c_str());
}

// keyword from string
inline cl_object Keyword(std::string const &k) {
  return ecl_make_keyword(util::upcase(k).c_str());
}

// quote -- useful for Eval(...) calls
inline cl_object Quote(cl_object sym) { return List(Symbol("QUOTE"), sym); }

// just for laughs
inline cl_object Car(cl_object cons) { return cl_car(cons); }
inline cl_object Cdr(cl_object cons) { return cl_cdr(cons); }
inline cl_object Cons(cl_object car, cl_object cdr) {
  return cl_cons(car, cdr);
}

// push element to list
inline void Push(cl_object element, cl_object &list) {
  list = Cons(element, list);
}

// non-destructively reverse list
inline void Reverse(cl_object &list) { list = cl_nreverse(list); }

//
// Converting C++ to Lisp
//

// integers
template <typename T,
          typename std::enable_if_t<std::is_integral<T>::value> * = nullptr>
inline cl_object toLisp(T num) {
  return ecl_make_integer(num);
}

// floats
inline cl_object toLisp(double num) { return ecl_make_double_float(num); }
inline cl_object toLisp(float num) { return ecl_make_single_float(num); }

// complex number
template <typename T,
          typename std::enable_if_t<std::is_arithmetic<
              std::remove_cv_t<std::remove_reference_t<T>>>::value> * = nullptr>
inline cl_object toLisp(std::complex<T> cplxf) {
  return ecl_make_complex(toLisp(cplxf.real()), toLisp(cplxf.imag()));
}

// strings
namespace util {
template <typename T> inline cl_object toLispString(T const &str) {
  // a string is just a lisp vector holding wide (4-byte) unicode characters
  auto const lvec = ecl_alloc_simple_vector(str.size(), ecl_aet_ch);
  std::size_t k = 0;
  for (auto const &c : str) {
    ecl_aset1(lvec, k++, ECL_CODE_CHAR(c));
  }
  return lvec;
}
} // namespace util

inline cl_object toLisp(std::string const &str) {
  return util::toLispString(str);
}
// need to explicitly specialize for string literals, they would otherwise be
// caught by the bool overload
inline cl_object toLisp(char const *str) {
  return util::toLispString(std::string(str)); // XXX this is not optimal
}

// booleans
inline cl_object toLisp(bool tf) { return tf ? t : nil; }

// sequences

namespace util {
// make a list from a C++ range -- easy
template <typename It> inline cl_object toLispList(It it, It const &stop) {
  auto list = List();
  for (; it != stop; ++it) {
    Push(toLisp(*it), list);
  }
  Reverse(list);
  return list;
}

// make a vector from a C++ range -- a little more involved

// automatically choose the right ECL element type when converting std::vectors
// and arrays
template <typename T> struct ECLElementType {
  static const cl_elttype type = ecl_aet_object; // default to 'object'
};

#define DEFINE_ECL_ELTYPE_CONVERSION(CTYPE, ECLTYPE)                           \
  template <> struct ECLElementType<CTYPE> {                                   \
    static const cl_elttype type = ECLTYPE;                                    \
  };

DEFINE_ECL_ELTYPE_CONVERSION(float, ecl_aet_sf);
DEFINE_ECL_ELTYPE_CONVERSION(double, ecl_aet_df);

DEFINE_ECL_ELTYPE_CONVERSION(std::int64_t, ecl_aet_fix);
DEFINE_ECL_ELTYPE_CONVERSION(std::uint64_t, ecl_aet_index);

DEFINE_ECL_ELTYPE_CONVERSION(std::uint8_t, ecl_aet_b8);
DEFINE_ECL_ELTYPE_CONVERSION(std::uint16_t, ecl_aet_b16);
DEFINE_ECL_ELTYPE_CONVERSION(std::uint32_t, ecl_aet_b32);
DEFINE_ECL_ELTYPE_CONVERSION(std::int8_t, ecl_aet_i8);
DEFINE_ECL_ELTYPE_CONVERSION(std::int16_t, ecl_aet_i16);
DEFINE_ECL_ELTYPE_CONVERSION(std::int32_t, ecl_aet_i32);

// given a cl_elttype type, make a lisp vector from the given c++ range
template <typename It, cl_elttype type>
inline cl_object toLispVector(It it, It const &stop) {
  auto lvec = ecl_alloc_simple_vector(stop - it, type);
  std::size_t k = 0;
  for (; it != stop; ++it) {
    ecl_aset1(lvec, k++, toLisp(*it));
  }
  return lvec;
}

// specialize for bit vector -- can create bit vectors from all integral types,
// but only if we know what we are doing...
template <typename T> cl_object toLispBitVector(T const &bits) {
  auto lvec = ecl_alloc_simple_vector(bits.size(), ecl_aet_bit);
  auto const one = toLisp(1);
  auto const zero = toLisp(0);
  for (std::size_t i = 0; i < bits.size(); ++i) {
    ecl_aset1(lvec, i, bits[i] ? one : zero);
  }
  return lvec;
}
} // namespace util

// specialize for the most-used sequence containers

// std::vector, std::deque, and std::array to lisp vectors
template <typename T> inline cl_object toLisp(std::vector<T> const &vec) {
  return util::toLispVector<
      decltype(cbegin(vec)),
      util::ECLElementType<typename std::decay<T>::type>::type>(cbegin(vec),
                                                                cend(vec));
}
template <typename T> inline cl_object toLisp(std::deque<T> const &vec) {
  return util::toLispVector<
      decltype(cbegin(vec)),
      util::ECLElementType<typename std::decay<T>::type>::type>(cbegin(vec),
                                                                cend(vec));
}
template <typename T, std::size_t N>
inline cl_object toLisp(std::array<T, N> const &array) {
  return util::toLispVector<
      decltype(cbegin(array)),
      util::ECLElementType<typename std::decay<T>::type>::type>(cbegin(array),
                                                                cend(array));
}

// std::list to a lisp list
template <typename T> inline cl_object toLisp(std::list<T> const &list) {
  return util::toLispList(cbegin(list), cend(list));
}

// vector<bool> and bitset -> bit vector
inline cl_object toLisp(std::vector<bool> const &vec) {
  return util::toLispBitVector(vec);
}
template <std::size_t N> inline cl_object toLisp(std::bitset<N> const &bits) {
  return util::toLispBitVector(bits);
}

// std::map and std::unordered_map -- convert to hash table
namespace util {
// some C++ types need 'equal as hash table test function. this requires some
// care.
template <typename T> struct ECLHashTableTest {
  static constexpr const char *test = "eql";
};

#define DEFINE_ECL_HT_TEST(CTYPE, TEST)                                        \
  template <> struct ECLHashTableTest<CTYPE> {                                 \
    static constexpr const char *test = TEST;                                  \
  };

// hash tables with string keys are compared with 'equal
DEFINE_ECL_HT_TEST(std::string, "equal");

template <typename Map> inline cl_object MapToHT(Map const &map) {
  auto ht = cl__make_hash_table(
      Symbol(ECLHashTableTest<typename Map::key_type>::test),
      ecl_make_fixnum(std::max((cl_fixnum)1024, (cl_fixnum)map.size())),
      cl_core.rehash_size, cl_core.rehash_threshold);
  for (auto const &kv : map) {
    si_hash_set(toLisp(kv.first), ht, toLisp(kv.second));
  }
  return ht;
}
} // namespace util

template <typename K, typename V>
inline cl_object toLisp(std::map<K, V> const &m) {
  return util::MapToHT(m);
}
template <typename K, typename V>
inline cl_object toLisp(std::unordered_map<K, V> const &m) {
  return util::MapToHT(m);
}

// pointers
template <typename T> inline cl_object CPointer(T *ptr) {
  return ecl_make_foreign_data(ECL_NIL, 0, ptr);
}

//
// Converting Lisp to C++
//

// there are some more conversion functions here:
// https://common-lisp.net/project/ecl/static/ecldoc/Standards.html#Numbers

inline std::string fromString(cl_object o) {
  switch (ecl_t_of(o)) {
  case t_string:
    if (!ecl_fits_in_base_string(o)) {
      o = cl_copy_seq(o);
    } else {
      o = si_copy_to_simple_base_string(o);
    }
    break;
  case t_base_string:
    break;
  default:
    return std::string(); // you get an empty string
  }
  // return std::string from the bytes of the lisp string
  return std::string(o->base_string.self,
                     o->base_string.self + o->base_string.dim);
}

// convert a symbols' string representation
inline std::string fromSymbol(cl_object o) {
  return fromString(o->symbol.name);
}

// use at own risk -- check with isByteVector first
inline std::vector<uint8_t> fromByteVector(cl_object o) {
  return std::vector<uint8_t>(o->vector.self.b8,
                              o->vector.self.b8 + o->vector.dim);
}

// numbers
inline int64_t fromInt(cl_object o) { return ecl_to_int64_t(o); }

inline uint64_t fromUint(cl_object o) { return ecl_to_uint64_t(o); }

inline float fromFloat(cl_object o) { return ecl_to_float(o); }
inline double fromDouble(cl_object o) { return ecl_to_double(o); }

inline std::complex<float> fromComplexFloat(cl_object o) {
  auto const re = fromFloat(Funcall(Symbol("realpart"), o));
  auto const im = fromFloat(Funcall(Symbol("imagpart"), o));
  return std::complex<float>(re, im);
}

inline std::complex<double> fromComplexDouble(cl_object o) {
  auto const re = fromDouble(Funcall(Symbol("realpart"), o));
  auto const im = fromDouble(Funcall(Symbol("imagpart"), o));
  return std::complex<double>(re, im);
}

// some simple sanity checks
inline bool isString(cl_object o) {
  auto const tt = ecl_t_of(o);
  return ((tt == t_string) || (tt == t_base_string));
}

inline bool isByteVector(cl_object o) {
  return ((ecl_t_of(o) == t_vector) && (o->vector.elttype == ecl_aet_b8));
}

//
// Sequence stuff
//

// apply fn to each element in a list or array. if seq is an atom, just apply
// the function to it
template <typename Fun> inline void forEach(cl_object seq, Fun fun) {
  if (cl_consp(seq) == t) {
    while (!Null(seq)) {
      fun(Car(seq));
      seq = Cdr(seq);
    }
  } else if (cl_arrayp(seq) == t) {
    auto const numel = fromInt(cl_array_total_size(seq));
    for (int64_t i = 0; i < numel; ++i) {
      fun(cl_row_major_aref(seq, toLisp(i)));
    }
  } else {
    fun(seq);
  }
}

// map fn over cl_list, returning the results in a flat std::vector
template <typename Fun>
inline std::vector<decltype(std::declval<Fun>()(std::declval<cl_object>()))>
map(cl_object const &seq, Fun fn) {
  decltype(map(seq, std::declval<Fun>())) o;
  forEach(seq, [&o, &fn](auto const &obj) { o.push_back(fn(obj)); });
  return o;
}

// filter -- predicate works on the lisp object, fn converts the lisp object to
// C++
template <typename Fun, typename Pred>
inline decltype(map(std::declval<cl_object>(), std::declval<Fun>()))
filter(cl_object const &seq, Fun fn, Pred pred) {
  decltype(map(seq, std::declval<Fun>())) o;
  forEach(seq, [&o, &fn, &pred](auto const &obj) {
    if (pred(obj)) {
      o.push_back(fn(obj));
    }
  });
  return o;
}

//
// Lisp "Environment"
//

class Environment {
public:
  // according to the ECL documentation, we need to initialize all modules in
  // order! since this header is meant to remain generic, we pass a vector of
  // the init function pointers as argument to the constructor.
  //
  // see:
  // https://common-lisp.net/project/ecl/static/ecldoc/Extensions.html#Build-it-as-static-library-and-use-in-C
  typedef void (*initializer)(cl_object);
  typedef std::vector<initializer> Initializers;

  // create the lisp env & initialize all modules
  static void init(Initializers const &initializers) {
    // disable asynchronous signal handlers
    // https://common-lisp.net/project/ecl/static/manual/re97.html
    constexpr int ECL_BOOL_FALSE = 0;
    ecl_set_option(ECL_OPT_TRAP_SIGSEGV, ECL_BOOL_FALSE);
    ecl_set_option(ECL_OPT_TRAP_SIGFPE, ECL_BOOL_FALSE);
    ecl_set_option(ECL_OPT_TRAP_SIGINT, ECL_BOOL_FALSE);
    ecl_set_option(ECL_OPT_TRAP_SIGILL, ECL_BOOL_FALSE);
    ecl_set_option(ECL_OPT_TRAP_SIGBUS, ECL_BOOL_FALSE);
    ecl_set_option(ECL_OPT_TRAP_SIGPIPE, ECL_BOOL_FALSE);
    ecl_set_option(ECL_OPT_TRAP_SIGCHLD, ECL_BOOL_FALSE);
    ecl_set_option(ECL_OPT_TRAP_INTERRUPT_SIGNAL, ECL_BOOL_FALSE);
    ecl_set_option(ECL_OPT_SIGNAL_HANDLING_THREAD, ECL_BOOL_FALSE);
    // fake some argv/argc and boot the lisp env (this actually segfaults if we
    // don't do this...)
    static auto av = "lisp";
    auto aav = const_cast<char *>(av);
    cl_boot(1, &aav);
    // initialize all modules
    for (auto const &i : initializers) {
      ecl_init_module(nullptr, i);
    }
  }

  // shut down the env
  static void shutdown() { cl_shutdown(); }

  // initializing & shutdown in object form, if you like that sort of thing
  Environment(Initializers const &i) { init(i); }
  ~Environment() { shutdown(); }
};

//
// LispThread --  Safe, single-threaded ECL execution
//

#if __has_include(<boost/asio.hpp>)
class LispThread {
public:
  // a blocking run(...) call that always executes on the lisp thread. need to
  // run EVERYTHING on this thread.
  template <typename Work> bool run(Work f) {
    bool ret = true;
    std::unique_lock<decltype(_mtx)> l(_mtx);
    _done = false;
    _io.dispatch([f, this, &ret] {
      std::unique_lock<decltype(_mtx)> l(_mtx, std::defer_lock);
      auto env = ecl_process_env();
      // execute any lisp code in protected region
      ECL_CATCH_ALL_BEGIN(env) { f(); }
      ECL_CATCH_ALL_IF_CAUGHT {
        // negative return value = something went wrong and we are restarting
        // the environment.
        ret = false;
        lisp::Environment::shutdown();
        lisp::Environment::init(_modInit);
        _initfn();
      }
      ECL_CATCH_ALL_END;
      l.lock();
      _done = true;
      l.unlock();
      _cv.notify_one();
    });
    _cv.wait(l, [this] { return _done; });
    return ret;
  }

  // an async run(...) call where you have to give a callback lambda. use at own
  // risk.
  template <typename Work, typename Callback>
  void async_run(Work f, Callback cb) {
    bool ret = true;
    _io.dispatch([f, cb, this, &ret] {
      auto env = ecl_process_env();
      ECL_CATCH_ALL_BEGIN(env) { f(); }
      ECL_CATCH_ALL_IF_CAUGHT {
        ret = false;
        lisp::Environment::shutdown();
        lisp::Environment::init(_modInit);
        _initfn();
      }
      ECL_CATCH_ALL_END;
      cb(ret);
    });
  }

  // tors

  typedef std::function<void()> Initfn;
  LispThread(Environment::Initializers modInit, Initfn initfn)
      : _iow(boost::asio::make_work_guard(_io)), _thread([this] { _io.run(); }),
        _modInit(modInit), _initfn(initfn) {
    _unsafe_run([this] {
      lisp::Environment::init(_modInit);
      _initfn();
    });
  }

  ~LispThread() {
    run([] { lisp::Environment::shutdown(); });
    _iow.reset();
    _io.stop();
    if (_thread.joinable()) {
      _thread.join();
    }
  }

private:
  // execution
  boost::asio::io_context _io;
  boost::asio::executor_work_guard<decltype(_io)::executor_type> _iow;
  std::thread _thread;
  std::condition_variable _cv;
  std::mutex _mtx;
  bool _done;

  // save copies of initializers
  Environment::Initializers _modInit;
  Initfn _initfn;

  // need this for the constructor
  template <typename Work> void _unsafe_run(Work f) {
    std::unique_lock<decltype(_mtx)> l(_mtx);
    _done = false;
    _io.dispatch([f, this] {
      std::unique_lock<decltype(_mtx)> l(_mtx, std::defer_lock);
      f();
      l.lock();
      _done = true;
      l.unlock();
      _cv.notify_one();
    });
    _cv.wait(l, [this] { return _done; });
  }
};

#endif // has_include(<boost/asio.hpp>)

} // namespace lisp

//
// Lispy methods on C++ classes
//

// the WITH_LISPY_METHODS macro is ripped from [1], it let's me access the
// typename of a class in a pointer cast within a static method.
//
// the OVERLOADED_MACRO is from [2] and does exactly what you think it does...
//
// [1] https://stackoverflow.com/questions/21143835
// [2] https://stackoverflow.com/questions/11761703
//
// clang-format off
#define WITH_LISPY_METHODS(X) X : public with_self_type<X>
#define WITH_LISPY_METHODS_DERIVED(X,...) X : public with_self_type<X,__VA_ARGS__>
template <typename... Ts> class with_self_type;
template <typename X, typename... Ts> class with_self_type<X, Ts...> : public Ts... {
protected:
  typedef X self;
};
#define OVERLOADED_MACRO(M, ...) _OVR(M, _COUNT_ARGS(__VA_ARGS__)) (__VA_ARGS__)
#define _OVR(macroName, number_of_args)   _OVR_EXPAND(macroName, number_of_args)
#define _OVR_EXPAND(macroName, number_of_args)    macroName##number_of_args
#define _COUNT_ARGS(...)  _ARG_PATTERN_MATCH(__VA_ARGS__, 9,8,7,6,5,4,3,2,1)
#define _ARG_PATTERN_MATCH(_1,_2,_3,_4,_5,_6,_7,_8,_9, N, ...)   N
// clang-format on
#define LISPY_METHOD(...) OVERLOADED_MACRO(LISPY_METHOD, __VA_ARGS__)
#define LISPY_METHOD3(NAME, PARAM_TYPE, CONVERSION)                            \
  void NAME(PARAM_TYPE data);                                                  \
  static cl_object NAME##_LISPMETHOD(cl_narg narg, ...) {                      \
    va_list args;                                                              \
    va_start(args, narg);                                                      \
    auto selfp = static_cast<self *>(                                          \
        ecl_foreign_data_pointer_safe(va_arg(args, cl_object)));               \
    auto data = CONVERSION(va_arg(args, cl_object));                           \
    selfp->NAME(data);                                                         \
    return ECL_T;                                                              \
  }
#define LISPY_METHOD1(NAME)                                                    \
  void NAME();                                                                 \
  static cl_object NAME##_LISPMETHOD(cl_narg narg, ...) {                      \
    va_list args;                                                              \
    va_start(args, narg);                                                      \
    auto selfp = static_cast<self *>(                                          \
        ecl_foreign_data_pointer_safe(va_arg(args, cl_object)));               \
    selfp->NAME();                                                             \
    return ECL_T;                                                              \
  }
#define LISPY_METHOD2(NAME, PARAM_TYPE)                                        \
  LISPY_METHOD3(NAME, PARAM_TYPE, PARAM_TYPE)

#endif // b00d9d87c4e5e57833e963e3b
