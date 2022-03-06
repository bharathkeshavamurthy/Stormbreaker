# script for automatically downloading and renaming database files
if [ -z "$1" ]
  then
    echo "No match number provided. Usage: $0 <match ID> <outputdir>"
    exit 1
fi

if [ -z "$2" ]
  then
    echo "No output dir provided. Usage: $0 <match ID> <outputdir>"
    exit 1
fi

ssh sc2-lz ls -d /share/nas/bamwireless/scrimmage$1_data/srn_logs/Team*/* > match_dir
sed -i -e 's/MATCH-/ /g' match_dir
sed -i -e 's/-RES/ /g' match_dir
awk '/bamwireless/ {print $2}' match_dir > match_num
rm match_dir*
homedir=$(pwd)
for num in $(cat match_num)
do
  echo "current match number: $num"
  mkdir $2/${num}
  cd $2/${num}
  ssh sc2-lz ls -d /share/nas/bamwireless/scrimmage$1_data/srn_logs/Team*/*MATCH-${num}-*/*srn*  > filenm
  sed -i -e 's/-srn/ /g' filenm
  sed -i -e 's/-/ /g' filenm
  awk '/bamwireless/ {print $6}' filenm > srnnum
  for number in $(cat srnnum)
  do
    scp sc2-lz:/share/nas/bamwireless/scrimmage$1_data/srn_logs/Team*/*MATCH-${num}-*/*srn${number}-*/log_sqlite.db ./srn${number}.db
  done
#  rm filenm*
  srns=$(cat srnnum)
  echo $srns
  cp $homedir/merge.py ./
  python3 merge.py out.db $srns
  cd $homedir
done
