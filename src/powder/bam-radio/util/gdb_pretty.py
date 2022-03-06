# gdb pretty printer for gnu radio tag_t types. this file will probably contain
# other things in the future as well.
#
# Copyright (c) 2017 Dennis Ogbe
#
# Use as follows:
#   (gdb) source gdb_pretty.py
# Then, just print like you normally would
#   (gdb) print tags

import sys
from subprocess import check_output
gcc_ver = check_output("gcc -dumpversion".split()).strip().decode()
sys.path.append("/usr/share/gcc-" + gcc_ver + "/python")
from libstdcxx.v6 import register_libstdcxx_printers

try:
    register_libstdcxx_printers(None)
except Exception as e:
    pass

# tag_t printer
class GRTagPrinter:
    def __init__(self, val):
        # try to find the write string function
        tmp = gdb.lookup_global_symbol("pmt::write_string")
        if tmp is None:
            tmp = gdb.lookup_global_symbol("pmt::write_string[abi:cxx11]")
        if tmp is None:
            tmp = gdb.lookup_global_symbol("pmt::write_string[abi:cxx11](boost::intrusive_ptr<pmt::pmt_base>)")
        # give up
        if tmp is None:
            raise gdb.GdbError("Could not find pmt::write_string function")
        # else get the strings
        self.fn = tmp.value()
        self.ofst = val["offset"]
        # get the std::basic_string objects
        key_val = self.fn(val["key"])
        self.key = self.get_string_from_string_val(key_val)
        value_val = self.fn(val["value"])
        self.value = self.get_string_from_string_val(value_val)

    # check out the definiiton of StdStringPrinter for these shenanigans
    def get_string_from_string_val(self, val):
        type = val.type
        new_string = type.name.find("::__cxx11::basic_string") != -1
        if type.code == gdb.TYPE_CODE_REF:
            type = type.target ()

        # Calculate the length of the string so that to_string returns
        # the string according to length, not according to first null
        # encountered.
        ptr = val ['_M_dataplus']['_M_p']
        if new_string:
            length = val['_M_string_length']
            # https://sourceware.org/bugzilla/show_bug.cgi?id=17728
            ptr = ptr.cast(ptr.type.strip_typedefs())
        else:
            realtype = type.unqualified ().strip_typedefs ()
            reptype = gdb.lookup_type (str (realtype) + '::_Rep').pointer ()
            header = ptr.cast(reptype) - 1
            length = header.dereference ()['_M_length']
        return ptr.string(length = length)

    # write the string to gdb prompt
    def to_string(self):
        print("gr::tag_t = {", end="")
        print("offset = {}, ".format(self.ofst), end="")
        print("key = {}, ".format(self.key), end="")
        print("value = {}".format(self.value), end="")
        print("}", end="")


# a pretty printer for any Qxx.16 fixed-point number
class Qx16Printer:
    def __init__(self, val):
        self.asFloat = float(int(val["rawVal"])) / float(1 << 16)
    def to_string(self):
        print(self.asFloat, end="")


def bam_debug_pretty_printers(val):
    if str(val.type) == "gr::tag_t":
        return GRTagPrinter(val)
    if str(val.type) == "tag_t":
        return GRTagPrinter(val)
    # this does not work for all Fp32 types, only the ones typedef'd to yaldpc
    # fixed (dividing by 1 << 16 above)
    if str(val.type) == "Fp::Fp32f<16u>":
        return Qx16Printer(val)
    if str(val.type) == "Fp::Fp64f<16u>":
        return Qx16Printer(val)


# register our lookup function
gdb.pretty_printers.append(bam_debug_pretty_printers)
