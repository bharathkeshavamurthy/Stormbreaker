#!/usr/bin/env python3

import scipy.io as sio
import numpy as np
import argparse
import os
import sys

# hack: need to get current directory on PATH to oad the proto definition
sys.path.append(os.getcwd())
import debug_pb2


def convert_file(filename):
    """convert the serialized protobuf message to .mat file"""
    fn = os.path.abspath(filename)
    root, ext = os.path.splitext(fn)
    bn = os.path.basename(filename)
    # parse protobuf
    fp = debug_pb2.DFTSOFDMFrameParams()
    with open(fn, "rb") as f:
        b = f.read()
        try:
            fp.ParseFromString(b)
        except:
            print("Could not parse {:s}. Skipping.".format(fn))
            return
    # convert to mat
    out = {}
    ## PARAMS
    for field in ('num_symbols', 'num_tx_samples', 'num_bits', 'dft_spread_length'):
        out[field] = getattr(fp, field)
    ## SYMBOLS
    # scalar fields
    sfields = [
        'symbol_length',
        'oversample_rate',
        'cyclic_prefix_length',
        'postfix_pad',
        'num_tx_samples',
        'num_data_carriers',
        'num_bits',
        'num_pilot_carriers'
    ]
    # carrier mappings (simple vector fields with 1+ re-indexing)
    svfields = ['data_carrier_mapping', 'pilot_carrier_mapping']
    # complex number fields
    cfields = ['pilot_symbols', 'prefix']
    s = list()
    for sym in fp.symbols:
        sl = {}
        for field in sfields:
            sl[field] = (getattr(sym, field))
        for field in svfields:
            sl[field] = (np.array(getattr(sym, field)) + 1)  # re-index for matlab
        for field in cfields:
            sl[field] = (np.array([z.re + 1j * z.im for z in getattr(sym, field)], dtype=np.complex64))
        s.append(sl)
    out["symbols"] = s
    # save to file
    outname = root + ".mat"
    sio.savemat(outname, out)
    print("Saved {:s}.".format(outname))


def main():
    """get a list of files from command line arguments"""
    parser = argparse.ArgumentParser(description="Get list of files")
    parser.add_argument("path", nargs="+", help="path of a file")
    args = parser.parse_args()

    # Parse paths
    full_paths = [os.path.join(os.getcwd(), path) for path in args.path]
    files = set()
    for path in full_paths:
        if os.path.isfile(path):
            files.add(path)

    # convert a file, assuming parsing it does not fail.
    for f in files:
        convert_file(f)


if __name__ == "__main__":
    main()
