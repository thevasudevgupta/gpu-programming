#!/bin/bash

#Do NOT Terminate terminate with a "/"
current_dir=$(pwd)
INPUT="$current_dir/Evaluation/Testcases/Input"
OUTPUT="$current_dir/Evaluation/Testcases/Output"

touch "A1-Marks.txt"
MARKSFILE="$current_dir/A1-Marks.txt"

echo "====================START==========================" >> $MARKSFILE
date >> $MARKSFILE

for FOLDER in SUBMIT/*
do
	#! echo FOLDER NAME : "${FOLDER}"
	cd "${FOLDER}"
	
	ROLLNO=$(ls *.cu | tail -1 | cut -d'.' -f1)
	LOGFILE=${ROLLNO}.log
	
	# check for single source file. If not halt script!
	if [ $(ls | wc -l) -ne 1 ] 
	then
		echo "May be cleanup files! (delete all files in 'A1-Submit/<ROLLNO>' folder except ROLLNO.cu) and run evaluate.sh!"
		break
	fi
	
	# cp require/our input files to stud's folder and build
	cp ${ROLLNO}.cu main.cu
	cp -r $INPUT .
	cp -r $OUTPUT .
	
	nvcc main.cu -o main
	
	# If build fails? then skip to next stud
	if [ $? -ne 0 ] 
	then
		echo $ROLLNO,BUILD FAILED!
    	echo $ROLLNO,BUILD FAILED! >> $MARKSFILE # write to file     
		cd ../.. # MUST
		continue
	fi
		
	date >> $LOGFILE

	for testcase in $INPUT/*; do 
		
		file_name=${testcase##*/}
		echo "$file_name"
		
		./main < $testcase
		diff -w "$OUTPUT/$file_name" "kernel1.txt" > /dev/null 2>&1
		exit_code=$?
		if (($exit_code == 0)); then
		  	echo "success" >> $LOGFILE
		else 
			echo "failure" >> $LOGFILE
		fi

		diff -w "$OUTPUT/$file_name" "kernel2.txt" > /dev/null 2>&1
		exit_code=$?
		if (($exit_code == 0)); then
		  	echo "success" >> $LOGFILE
		else 
			echo "failure" >> $LOGFILE
		fi

		diff -w "$OUTPUT/$file_name" "kernel3.txt" > /dev/null 2>&1
		exit_code=$?
		if (($exit_code == 0)); then
		  	echo "success" >> $LOGFILE
		else 
			echo "failure" >> $LOGFILE
		fi


	done
  
	SCORE=$(grep -ic success $LOGFILE) #Counts the success in log
	#! TOTAL=$(ls $INPUT/*.txt | wc -l)
	echo $ROLLNO,$SCORE 
	echo $ROLLNO,$SCORE >> $MARKSFILE # write to file 
	
	# IMPORTANT
	cd ../..
done

date >> $MARKSFILE
echo "====================DONE!==========================" >> $MARKSFILE

