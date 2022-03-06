#!/bin/sh

set -e

grep -a "^frame SNR" out | awk '{print $3;}' >| snr.out
grep -a "^triggeryum" out | awk '{print $2;}' >| triggery
grep -a "^triggerbum" out | awk '{print $2;}' >| triggerb
grep -a "^triggerbum" out | awk '{print $4;}' >| triggert
grep -a "^triggerbum" out | awk '{print $6;}' >| triggers
grep -a "^triggerbum" out | awk '{print $8;}' >| triggern
grep -a "^triggerbum" out | awk '{print $10;}' >| triggerw
grep -a "^triggerbum" out | awk '{print $12;}' >| triggera
grep -a "^detect_preamble" out | awk '{print $7;}' >| dpnir 
grep -a "^detect_preamble" out | awk '{print $5;}' >| dpN
grep -a "^detect_preamble" out | awk '{print $3;}' >| dpn
