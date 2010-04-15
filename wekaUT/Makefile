# Java version to use
JAVADIR = /home/backup/lib/j2sdk1.4.2_19

# a location where a temporary directory will be created and the compilation will take place
TMP_LOCATION = /var/local/weka-latest

# a directory where a successful build should be copied
PERM_LOCATION = /u/ml/software/weka-latest

# name of the file that will contain all source files
SOURCE_FILE_LIST = sourcefiles.log

# name of the file that will contain all class file names
CLASS_FILE_LIST = classfiles.log

# name of the file where the compilation log will be stored
CHECKOUT_LOG_FILE = checkout.log

# name of the file where the compilation log will be stored
COMPILE_LOG_FILE = compile.log

# name of the file where the javadoc log will be stored
JAVADOC_LOG_FILE = javadoc.log

# person to receive the email
RECIPIENT = mbilenko
CLASSPATH = ../packages/rmn.jar:../packages/Jama-1.0.1.jar:../packages/cplex.jar:../packages/jmage.jar:../packages/ir.jar:../packages/jmatlink.jar:../packages/colt-1.2.0.jar

current:
	find ./ -path './*.java' > ${SOURCE_FILE_LIST} ;\
	${JAVADIR}/bin/javac -source 1.4 -O -classpath ${CLASSPATH} @${SOURCE_FILE_LIST}

# Create weka jar file
jar:
	jar cvf weka.jar weka


# Create jar file with all the class files
classjar:
	find ./ -path './*.class' > ${CLASS_FILE_LIST} ;\
	jar cvf weka-all-class.jar @${CLASS_FILE_LIST}

remove: 
	rm -f ${SOURCE_FILE_LIST}

javadoc:
		rm -rf doc/*;
		javadoc -d doc -classpath ${CLASSPATH} @${SOURCE_FILE_LIST}
