#!/lusr/bin/perl 


# Perl code to create a gnuplot file on the fly from experimenter
# (c) 2003 Mikhail Bilenko 

open(FILE, "<$ARGV[0]");
@data = <FILE>;

my @curveNames;
my @curves;

my %nameNamesHash;
my %nameCurvesHash; 


($numResultsets) = $data[2] =~ /(\d+)/;
print "$numResultsets results\n";

my $i;
my $num1;
my $num2;
$baselineProcessed = 0;
for ($i = 5; $i <= $#data; $i++) {
    $line = $data[$i];

    if ($line =~ /Comparing/) {
	($num1, $num2) = $line =~ /.+\((\d+)\).+\((\d+)\)/;
	print "1 = $num1, 2 = $num2\n";
    }

    
    if ($line =~ /Points/) {
	(@xpoints) = $line =~/(\d+.?\d*)/g;
	
	# lines after __ contain data
	$i = $i+2;
	while (substr($data[$i],0,1) =~/\S/) {
	    ($name) = $data[$i] =~ /(\S+)/;
	    print "Name: $name \n";

	    if ($numResultsets > 1) {
		# we have multiple resultsets

		# cut off name and final parens
		($line) = $data[$i] =~ /\S+(.+)\(.+\)/;

		(@cols) = split(/[ v*\/]+/,$line);

		@ypoints = ();
		@base = ();

		for ($l = 0; $l < $#cols; $l++) {
		    $ypoints[$l/2] = $cols[$l];
		    $base[$l/2] = $cols[$l+1];
		}

		if ($baselineProcessed eq 0) {
		    $basename = $name . "-" . $num2;
		    push(@curveNames, $basename);
		    push(@curves, [@base]);
		    print "Adding $#base\n";

		    @names = ();
		    push(@names, $basename);
		    $nameNamesHash{$name} = [@names];
		    @nameCurves = ();
		    push (@nameCurves, [@base]);
		    $nameCurvesHash{$name} = [@nameCurves];
		    
		}

		$curvename = $name . "-" . $num1;
		push(@curveNames, $curvename);
		push(@curves, [@ypoints]);

		if (exists($nameNamesHash{$name})) {
		    @nameNames =  @{$nameNamesHash{$name}};
		    push(@nameNames, $curvename);
		    $nameNamesHash{$name} = [@nameNames];
		    
		    @nameCurves = @{$nameCurvesHash{$name}};
		    push (@nameCurves, [@ypoints]);
		    $nameCurvesHash{$name} = [@nameCurves];
		}
	    } else {
		# we have a single result set
		($line) = $data[$i] =~ /\S+(.+)/;
		(@ypoints) = $line =~/([\d.]+)/g;
		# pad missing values with 0's
		if ($#ypoints < $#xpoints) {
		    for ($k = $#ypoints+1; $k <= $#xpoints; $k++) {
			$ypoints[$k] = 0;
		    }
		} 
		push(@curveNames, $name);
		push(@curves, [@ypoints]);
		print "Adding $#ypoints\n";
	    }
	    $i = $i + 1;
	}
	if ($baselineProcessed eq 0) {
	    $baselineProcessed = 1;
	}
    }
}



#individual files for different datasets

if ((keys %nameNamesHash) > 1) {
    foreach $name (keys %nameNamesHash) {

	$dataFileName = $ARGV[0]."-".$name.".dat";
	$gnuplotFileName = $ARGV[0]."-".$name.".gplot";

	@nameNames = @{$nameNamesHash{$name}};
	@nameCurves = @{$nameCurvesHash{$name}};

	open (DATAFILE, ">$dataFileName");

	for ($i = 0; $i <= $#xpoints; $i++) {
	    print DATAFILE $xpoints[$i];
	    for ($j = 0; $j <= $#nameCurves; $j++) {
		print DATAFILE "\t $nameCurves[$j][$i]";
	    }
	    print DATAFILE "\n";
	} 


	open (GNUPLOTFILE, ">$gnuplotFileName");

	print GNUPLOTFILE "#set terminal postscript eps 16\n";
	print GNUPLOTFILE "#set output \"/u/mbilenko/r/fig/05-jmlr/$name.eps\"\n\n";
	print GNUPLOTFILE "set key bottom right box\n\n";
#	print GNUPLOTFILE "set key 400, 0.5\n\n";

	print GNUPLOTFILE "set xlabel \"Number of Constraints\"\n";
#	print GNUPLOTFILE "set ylabel \"F-Measure\"\n";
	print GNUPLOTFILE "set ylabel \"NMI\"\n";

	print GNUPLOTFILE "set data style linespoints\n";
	print GNUPLOTFILE "plot ";
	for ($i = 0; $i < $#nameNames; $i++) {
	    $col = $i + 2;
	    print GNUPLOTFILE " \'$dataFileName\' using 1:$col title \"$nameNames[$i]\", ";
	}
	$col = $#nameNames + 2;

	print GNUPLOTFILE " \'$dataFileName\' using 1:$col title \"$nameNames[$#nameNames]\"\n";
	close GNUPLOTFILE;
	close DATAFILE;
	close FILE;
    }

} else {   # just a single output file
    $dataFileName = $ARGV[0].".dat";
    $gnuplotFileName = $ARGV[0].".gplot";

    open (DATAFILE, ">$dataFileName");

    for ($i = 0; $i <= $#xpoints; $i++) {
	print DATAFILE $xpoints[$i];
	for ($j = 0; $j <= $#curves; $j++) {
	    print DATAFILE "\t $curves[$j][$i]";
	}
	print DATAFILE "\n";
    } 


# overall file


    open (GNUPLOTFILE, ">$gnuplotFileName");

    print GNUPLOTFILE "\#set terminal postscript eps 16\n";
    print GNUPLOTFILE "\#set output \"filename.eps\"\n\n";
    print GNUPLOTFILE "\#set key bottom left box\n\n";

    print GNUPLOTFILE "\#set xlabel \"Recall\"\n";
    print GNUPLOTFILE "\#set ylabel \"Precision\"\n";

    print GNUPLOTFILE "set data style linespoints\n";
    print GNUPLOTFILE "plot ";
    for ($i = 0; $i < $#curveNames; $i++) {
	$col = $i + 2;
	print GNUPLOTFILE " \'$dataFileName\' using 1:$col title \"$curveNames[$i]\", ";
    }
    $col = $#curveNames + 2;

    print GNUPLOTFILE " \'$dataFileName\' using 1:$col title \"$curveNames[$#curveNames]\"\n";

    print GNUPLOTFILE "# pause -1\n";
    close GNUPLOTFILE;
    close DATAFILE;
    close FILE;
}







