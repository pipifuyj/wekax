#!/usr/bin/php
<?php
	$total = 0;
	$diff = 0;
	$file1 = $argv[1];
	$file2 = $argv[2];
	$predict1 = file($file1,FILE_IGNORE_NEW_LINES);
	$predict2 = file($file2,FILE_IGNORE_NEW_LINES);
	if(count($predict1) !== count($predict2)) exit(count($predict1)."!=".$predict2);
	echo count($predict1).count($predict2);
	for($i = 0; $i < count($predict1); $i ++){
		$line1 = preg_split("/\s+/",trim($predict1[$i]));
		$line2 = preg_split("/\s+/",trim($predict2[$i]));
		$a = intval($line1[1]);
		$p1 = doubleval($line1[2]);
		$b = intval($line2[1]);
		$p2 = doubleval($line2[2]);	
		if( abs($a*$p1 - $b*$p2) > 1) $diff ++;
		$total ++;
	}
	echo "总个数：".$total."\n";	
	echo "相异个数：".$diff."\n";
	echo "相异概率：".$diff/$total;
?>
