# ** Change this to your wekaUT install directory **
INSTALL_DIR = /home/backup/svn/wekax.googlecode.com/wekaUT

# Java version to use
JAVADIR = /home/backup/lib/j2sdk1.4.2_19

# Checkout name
VERSION_TO_CHECKOUT = weka335

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
CLASSPATH = ./:../:../../:../../../:${INSTALL_DIR}/packages/rmn.jar:${INSTALL_DIR}/packages/Jama-1.0.1.jar:${INSTALL_DIR}/packages/cplex.jar:${INSTALL_DIR}/packages/jmage.jar:${INSTALL_DIR}/packages/ir.jar:${INSTALL_DIR}/packages/jmatlink.jar

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

# Complete recompilation from a CVS snapshot
all:  checkout compile wrapup 

# checkout the latest code from the repository into a temporary directory
checkout:
	rm -rf ${TMP_LOCATION};\
	mkdir ${TMP_LOCATION};\
	cd ${TMP_LOCATION};\
	cvs checkout ${VERSION_TO_CHECKOUT}  >${CHECKOUT_LOG_FILE} 2>&1

# Compile all the source files
compile:
	cd ${TMP_LOCATION};\
	find ./ -iname *.java > ${SOURCE_FILE_LIST} ;\
	${JAVADIR}/bin/javac -source 1.4 -O -classpath ${CLASSPATH} @${SOURCE_FILE_LIST} >${COMPILE_LOG_FILE} 2>&1


# If errors were encountered, send email
# otherwise, copy a successful compilation into the permanent location
# and create documentation

wrapup:   

	if grep -s 'error' ${TMP_LOCATION}/${COMPILE_LOG_FILE}; then \
		mail -s 'Weka compilation broken!' ${RECIPIENT} < ${TMP_LOCATION}/compile.log; \
	else \
		rm -rf ${PERM_LOCATION}/*;\
		cp -R ${TMP_LOCATION}/${VERSION_TO_CHECKOUT}/* ${PERM_LOCATION};\
		cd ${PERM_LOCATION};\
		${JAVADIR}/bin/javac -source 1.4 -O -classpath ${CLASSPATH} ${PERM_LOCATION}/GetAllSubPackages.java;\
		cd ${PERM_LOCATION};\
		${JAVADIR}/bin/java GetAllSubPackages packages.txt ${PERM_LOCATION};\
		rm -rf ${PERM_LOCATION}/doc/*;\
		${JAVADIR}/bin/javadoc -source 1.4 -d ${PERM_LOCATION}/doc -classpath ${CLASSPATH} @packages.txt  >${JAVADOC_LOG_FILE} 2>&1;\
		chmod -R g+w *;\
	fi;

javadoc:
		rm -rf doc/*;
		javadoc -d doc -classpath ${CLASSPATH} @${SOURCE_FILE_LIST}
