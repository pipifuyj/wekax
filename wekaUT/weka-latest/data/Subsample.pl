#!/usr/bin/perl
# Given an arff file, extract a random subsample
# of data from it


# USAGE:   Subsample.pl FILENAME.ARFF NEWNAME PROPORTION
# e.g. Subsample.pl iris.arff iris15 0.1  produces iris2.arff, containing 0.1 of original data

#  OR

# Subsample.pl digits.arff digits3 0.5 1 2 3
# produces digits3.arff, containing 0.1 of original instances that belong to one of the classes 1, 2, 3

$ARFF_FILE = shift;
$OUT_FILE = shift;
$proportion = shift;

@classes = @ARGV;

# index each class by its idx+1
@class_hash{@classes} = (1 .. @classes) ;

    
my %data; 
open(INPUT, $ARFF_FILE) or die "Bad file: $!\n";
my @lines = <INPUT>;
close(INPUT);


open (OUTPUT, ">$OUT_FILE.arff");


$i = 0;
$pastHeader = 0;
my @data;
foreach $line(@lines) {
    chomp $line;
    if ($pastHeader < 1) {

	if ($line =~ /@(data)/i){
	    $pastHeader = 1;
	}

	# do we care about classes?
	if ($#classes < 0) {   # get all classes
	    print OUTPUT "$line\n";
	} else {               # we only want specific classes
	    if  ($line =~ /@(attribute (\')?class)/i){
		print OUTPUT "\@attribute class {";
		print OUTPUT join(", ", @classes);
		print OUTPUT "}\n";
	    } else {
		print OUTPUT "$line\n";
	    }
	} 
    } else {   # we are past header

	if ($#classes < 0)  { # we get all classes
	    if (length($line) > 0) {
		push(@data, $line);
	    }
	} else {              # we need to filter for only those classes that we care about
	    if (length($line) > 0) {
		(@attributes) = split(/(\s)*,(\s)*/,$line);
		$class_value = $attributes[$#attributes];
		if ($class_hash{$class_value} > 0) {  # we care about this one
		    push(@data, $line);
		} 
	    }	
	} 
    }
}

$permutations = factorial( scalar @data );
@shuffle = @data [ n2perm( 1+int(rand $permutations), $#data ) ];

print "The new dataset will have " . $proportion*(1+$#shuffle) . " instances\n";

for ($i=0; $i<($proportion * (1+$#shuffle)); ++$i) {
    print OUTPUT "$shuffle[$i]\n";
}

close OUTPUT;

# pat2perm(@pat) : turn pattern returned by n2pat() into
# permutation of integers.  XXX: splice is already O(N)
sub pat2perm {
    my @pat    = @_;
    my @source = (0 .. $#pat);
    my @perm;
    push @perm, splice(@source, (pop @pat), 1) while @pat;
    return @perm;
}

# n2perm($N, $len) : generate the Nth permutation of $len objects
sub n2perm {
    pat2perm(n2pat(@_));
}

BEGIN{
    my @fact = (1);
    sub factorial($) {
	my $n = shift;
	return $fact[$n] if defined $fact[$n];
	$fact[$n] = $n * factorial($n - 1);
    }
}


sub n2pat {
    my $i   = 1;
    my $N   = shift;
    my $len = shift;
    my @pat;
    while ($i <= $len + 1) {   # Should really be just while ($N) { ...
        push @pat, $N % $i;
        $N = int($N/$i);
        $i++;
    }
    return @pat;
}

sub commify_series {
    (@_ == 0) ? ''                                      :
    (@_ == 1) ? $_[0]                                   :
    (@_ == 2) ? join(" and ", @_)                       :
                join(", ", @_[0 .. ($#_-1)], "and $_[-1]");
}
