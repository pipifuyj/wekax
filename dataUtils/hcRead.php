<?php
/**
 * Read $hc(file) to $hc(data)
 */
if(!$usage)exit("No Usage!\n");
$hc=file($hc,FILE_IGNORE_NEW_LINES|FILE_SKIP_EMPTY_LINES);
foreach($hc as &$_hc){
	preg_match("/^(.+)\((.+)\)/",$_hc,$_hc);
	$_items=preg_split("/\s+/",trim($_hc[1]));
	list($_support,$_confidence)=preg_split("/\s+/",trim($_hc[2]));
	if($_pos=strpos($_support,"%"))$_support=substr($_support,0,$_pos)/100;
	if($_pos=strpos($_confidence,"%"))$_confidence=substr($_confidence,0,$_pos)/100;
	$_hc=array($_items,$_support,$_confidence);
	unset($_hc);
}
?>
