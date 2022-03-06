# script for automatically downloading and renaming database files
if [ -z "$1" ]
  then
    echo "No Reservation ID provided. Usage: $0 <Job Reservation ID> <outputdir>"
    exit 1
fi

if [ -z "$2" ]
  then
    echo "No job directory provided. Usage: $0 <Job Reservation Directory> <outputdir>"
    exit 1
fi



ssh sc2-lz ls -d /share/nas/bamwireless/RESERVATION-$1/bamwireless*srn*/ > filenm
sed -i -e 's/-srn/ /g' filenm
sed -i -e 's/-/ /g' filenm
awk '/bamwireless/ {print $5}' filenm > srnnum
for number in $(cat srnnum)
do
  scp sc2-lz:/share/nas/bamwireless/RESERVATION-$1/bamwireless*srn${number}-*/log_sqlite.db $2/srn${number}.db
done
rm filenm*
srns=$(cat srnnum)
echo $srns
python3 merge.py out.db $srns
