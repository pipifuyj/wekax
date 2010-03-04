#!/usr/bin/php
<?php
file_put_contents($argv[1].".qr.prg","");
$num = 1;
for($i = 0; $i < $argv[1]; $i ++){
	$file1 = $i.".sample";
	$file2 = $i.".label"; 
	$file1 = fopen($file1, "r") or exit("Unable to open file!");
	$file2 = fopen($file2, "r") or exit("Unable to open file!");
	while(!feof($file1))
	{
		$line = fgets($file1);
		if("" !== $pieces[0]) $QueryResult[$num][0] = intval($line);

		$line = fgets($file2);
		if("" !== $pieces[0]) $QueryResult[$num][1] = intval($line);	
		
		if($QueryResult[$num][1] === 1 || $QueryResult[$num][1] === -1){
			file_put_contents($argv[1].".qr.prg",$QueryResult[$num][0]." ". $QueryResult[$num][1]."\n",FILE_APPEND);		
			$num ++;
		}
	}
}
?>
