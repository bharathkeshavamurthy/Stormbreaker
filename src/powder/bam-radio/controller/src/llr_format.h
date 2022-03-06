// -*-c++-*-
// Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
//
// one typedef for the current format of the LLRs

#ifndef bed509828f52ee8013416ebb4e67ed00f557a335
#define bed509828f52ee8013416ebb4e67ed00f557a335

#include "bam_constellation.h"

namespace bamradio {
//
// if we want to change the LLR type, we comment and uncomment the desired
// type below
//
// typedef yaldpc::fixed llr_type;
// typedef yaldpc::fixed64 llr_type;
typedef float llr_type;

//
// anywhere where we are dealing with constellation object sptrs, use this
// typedef
//
namespace constellation {
typedef base<llr_type>::sptr sptr;
}
}

#endif // bed509828f52ee8013416ebb4e67ed00f557a335 //
