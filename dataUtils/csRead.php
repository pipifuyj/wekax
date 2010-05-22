<?php
/**
 * Read $cs(file) to $cs(data)
 */
if(!$usage)exit("No Usage!\n");
$_cs=file($cs,FILE_IGNORE_NEW_LINES|FILE_SKIP_EMPTY_LINES);
$cs=array();
foreach($_cs as $__cs){
	$__cs=preg_split("/\s+/",trim($__cs));
	$cs[$__cs[0]]=array($__cs[1],$__cs[2]);
}
?>
