#!/usr/bin/perl
use Getopt::Std;

getopt("io");

if (!$opt_i) {
	print "USAGE: CCS2Sparse.pl [switches]
	\t-i prefix for input files
	\t[-o output file]\n\n";
	exit(0);
}

print "Reading in files...\n";

# have to open _dim, _col_ccs, _row_ccs, _tfn_nz
print "\tdimension file\n";

$if = $opt_i."_dim";
open(INF,$if) or die ("Error: $!\n");
@ary = <INF>;
close(INF);

$temp = $ary[0];
chomp($temp);
($m, $n, $nz) = split(/ /,$temp);

print "\tcolumn ptr file\n";
$if = $opt_i."_col_ccs";
open(INF,$if) or die ("Error: $!\n");
@ary = <INF>;
close(INF);

$ctr = 0;
@colptrs = ();
foreach $line(@ary) {
	chomp($line);
	$colptrs[$ctr] = $line;
	$ctr += 1;
}

print "\trow index file\n";
$if = $opt_i."_row_ccs";
open(INF,$if) or die ("Error: $!\n");
@ary = <INF>;
close(INF);

$ctr = 0;
@rowindices = ();
foreach $line(@ary) {
	chomp($line);
	$rowindices[$ctr] = $line;
	$ctr += 1;
}

print "\tdata file\n";
$if = $opt_i."_tfn_nz";
open(INF,$if) or die ("Error: $!\n");
@ary = <INF>;
close(INF);

$valctr = 0;
@values = ();
foreach $line(@ary) {
	chomp($line);
	$values[$valctr] = $line;
	$valctr += 1;
}

if ($valctr != $ctr) {
	print "Error in input file - insufficient entries in values/rowindices\n";
	exit(0);
}

if ($opt_p) {
	print "\nOriginal values\n";
	print "\t$line1\n";
	print "\t@colptrs\n";
	print "\t@rowindices\n";
	print "\t@values\n\n";
}

print "Computing original matrix...\n";

# the i-th column starts at values(colptrs(i)) and ends at values(colptrs(i+1)-1).
# rowindices[colptrs[i] gives the row index of teh first nonzero element
# val[colptrs[i]] gives the value of the first nonzero element
%val = ();

for ($i = 0; $i < $n; $i += 1) { # runs through the columns
	#now run through the rows in that colums
	for ($j = $colptrs[$i]; $j < $colptrs[($i+1)]; $j += 1) {
		# the position of each element is rowindices[$j],$i
		$val{$rowindices[$j] + 1}{$i + 1} = $values[$j];
	}
}

if (!$opt_o) {
	$outfile = $opt_i."_fromCCS";
}
else {
	$outfile = $opt_o;
}
print "Writing sparse data to $outfile...\n";

open(OUTF,">$outfile");
	
foreach $key(keys %val) {
	foreach $innerkey(keys%{$val{$key}}) {
		print OUTF "$key $innerkey $val{$key}{$innerkey}\n";
	}
}

close(OUTF);

open(SPARSE,"$outfile");
open (FINAL, ">$outfile.arff");

$maxAttribute = -1;
while(<SPARSE>) {
    chomp;
    ($attribute, $instance, $value) = split(/ /);
    if ($attribute > $maxAttribute) {
	$maxAttribute = $attribute;
    }
    $instanceHash[$instance-1]{$attribute-1} = $value; # attributes and instances start from 1
}

$numInstances = scalar @instanceHash; 
$classAttrib = $maxAttribute;

open(PAT,"<$opt_i"."_label");
$rowNum = 0;
$maxClassValue=-1;
while (<PAT>) {
    chomp;
    $value = $_;
    if ($maxClassValue < $value) {
	$maxClassValue = $value;
    }
    $pattern{$rowNum} = $value;
    $rowNum++;
}

print FINAL "\@relation $opt_i\n\n";
$wf = $opt_i."_words";
open(WORDS,$wf) || die "Could not open word file!!";
$maxWords = <WORDS>;
chomp($maxWords);
print "NumAttributes: $maxWords, NumInstances: $numInstances, MaxClassIndex: $maxClassValue\n";

for ($num = 0; $num < $maxWords; $num++) {
    $word = <WORDS>;
    chomp($word);
    print FINAL "\@attribute $word numeric\n";
}
print FINAL "\@attribute _class_ {";
for ($num=0; $num<$maxClassValue; $num++) {
    print FINAL "$num, ";
}
print FINAL "$num}\n";

print FINAL "\n\@data\n";
for ($num = 0; $num < $numInstances; $num++) {
    print FINAL "{";
    @sortedKeys = sort numerically (keys %{$instanceHash[$num]});
    $lengthSortedKeys = scalar @sortedKeys;
    for ($keynum = 0; $keynum < $lengthSortedKeys - 1; $keynum++) {
	$key = $sortedKeys[$keynum];
	print FINAL "$key $instanceHash[$num]{$key}, ";
    }
    $key = $sortedKeys[$keynum];
    print FINAL "$key $instanceHash[$num]{$key}";

    if ($pattern{$num} != 0) {
	print FINAL ", $classAttrib $pattern{$num}}\n";
    }
    else {
	print FINAL "}\n";
    }
}

system("rm $outfile");


print "Done\n";

sub numerically {$a <=> $b;}
