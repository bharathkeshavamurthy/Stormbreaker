//  Copyright © 2018 Stephen Larew

#ifdef BAMASSERT_H
#undef BAMASSERT_H
#undef bamprecondition
#undef bamassert
#endif
#define BAMASSERT_H

#include "events.h"

#ifdef NDEBUG
#ifndef NOBAMASSERT
#define NOBAMASSERT
#endif
#endif

#ifdef NOBAMASSERT
#define bamassert(expr) (static_cast<void>(0))
#else
#define bamassert(expr)                                                        \
  do {                                                                         \
    if (!(expr)) {                                                             \
      using namespace std::string_literals;                                    \
      ::bamradio::log::doomsday("assert failed: "s + #expr, __FILE__,          \
                                __LINE__, __PRETTY_FUNCTION__);                \
    }                                                                          \
  } while (false)
#endif

#ifdef NOBAMPRECONDITION
#define bamprecondition(expr) (static_cast<void>(0))
#else
#define bamprecondition(expr)                                                  \
  do {                                                                         \
    if (!(expr)) {                                                             \
      using namespace std::string_literals;                                    \
      ::bamradio::log::doomsday("precondition failed: "s + #expr, __FILE__,    \
                                __LINE__, __PRETTY_FUNCTION__);                \
    }                                                                          \
  } while (false)
#endif
