1. Download the databases Uniref50 and Uniclust30 from databases.txt
2. Extract
3. Convert Uniref from XML to FASTA with (30162111 entries)
	python convert.py uniref50.xml
4. Install and compile HHSuite from https://github.com/soedinglab/hh-suite 
	very good userguide avaiable under https://github.com/soedinglab/hh-suite/wiki
	but installation instruction from readme.md works better!
5. Test with longest sequence in uniref50 (Titin) approx. 150min
	hhblits -cpu 4 -i testsequ/titin.sequ -d <path to uniclust30> -oa3m titin.a3m -maxres 50000 -n 3
6. Test with smallest sequence in uniref50 (ATP binding) approx. 35s 
	hhblits -cpu 4 -i testsequ/atpbind.sequ -d <path to uniclust30> -oa3m atpbind.a3m -maxres 50000 -n 3
7. Test creating of an alignment DB
	i) Creating database from FASTA (eg. ffindex_from_fasta -s testsequ/testdb.ff{data,index} testsequ/testdb.fasta) 
	ii) Create Database of MSAs with hhblits approx. 11min
		mpirun -np <num_threads> hhblits_mpi -i testsequ/testdb -d <path to uniclust30> -oa3m <testdb>_a3m -n 3 -cpu 1 -v 0
