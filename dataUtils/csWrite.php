<?php
/**
 * Write $cs to stdout
 */
class csWrite{
	public static function write($index,$cs){
		echo $index," ",implode(" ",$cs),"\n";
	}
	public static function output($CS){
		foreach($CS as $index=>$cs)self::write($index,$cs);
	}
}
if(isset($cs))csWrite::output($cs);
?>
