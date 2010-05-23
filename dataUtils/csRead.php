<?php
/**
 * Read $cs(file) to $cs(data)
 */
if(!$usage)exit("No Usage!\n");
if(!isset($csRead))$csRead=array('class'=>array());
$_cs=file($cs,FILE_IGNORE_NEW_LINES|FILE_SKIP_EMPTY_LINES);
$cs=array();
foreach($_cs as $__cs){
	$__cs=preg_split("/\s+/",trim($__cs));
	$cs[array_shift($__cs)]=$__cs;
	array_push($csRead['class'],$__cs[0],$__cs[2]);
}
sort($csRead['class']=array_unique($csRead['class']));
$csRead['number']=count($csRead['class']);
?>
