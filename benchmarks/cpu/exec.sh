cd $1
pytest ./benchmarks.py --benchmark-save="data" --benchmark-sort=name --benchmark-min-rounds=20
cd ../