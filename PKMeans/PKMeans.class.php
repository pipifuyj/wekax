<?php
class PKMeans{
	public $data=array();
	public $k=0;
	
	public $n=0;
	
	public $means=array();
	
	public $p=array();
	public $clusters=array();
	
	public $sims=array();
	
	public function PKMeans($data,$k){
		$this->data=$data;
		$this->k=$k;
		$this->n=count($data);
	}
	
	function sim($i,$j){
		if($this->sims[$i][$j]!==null)return $this->sims[$i][$j];
		$a=$this->data[$i];
		$b=$this->data[$j];
		if(($n=count($a))==count($b)){
			$AB=0;
			$AA=0;
			$BB=0;
			for($i=0;$i<$n;$i++){
				$AB+=$a[$i]*$b[$i];
				$AA+=$a[$i]*$a[$i];
				$BB+=$b[$i]*$b[$i];
			}
			$this->sims[$i][$j]=$AB/sqrt($AA*$BB);
		}else{
			$this->sims[$i][$j]=null;
			exit("Error with calculate sim between $i and $j\n");
		}
		return $this->sims[$i][$j];
	}
	
	public function crfun($cluster,$m){
		$n=count($cluster);
		$cr=0;
		for($i=0;$i<$n;$i++){
			$cr+=$this->sim($cluster[$i],$m);
		}
		$cr=sqrt($cr);
		return $cr;
	}
	
	public function initMeans(){
		$this->means=array();
		for($i=0;$i<$this->k;$i++){
			$this->means[]=$i;
		}
	}
	
	public function getMean($cluster){
		$n=count($cluster);
		$m=0;
		$cr=$this->crfun($cluster,$cluster[$m]);
		for($i=1;$i<$n;$i++){
			$t=$this->crfun($cluster,$cluster[$i]);
			if($t>$cr){
				$m=$i;
				$cr=$t;
			}
		}
		return $cluster[$m];
	}
	
	public function setMeans(){
		for($i=0;$i<$this->n;$i++){
			$p=0;
			$d=$this->sim($i,$this->means[$p]);
			for($j=1;$j<$this->k;$j++){
				$t=$this->sim($i,$this->means[$j]);
				if($t>$d){
					$p=$j;
					$d=$t;			}
			}
			$this->p[$i]=$p;
		}
		$this->clusters=array();
		for($i=0;$i<$this->n;$i++){
			$this->clusters[$this->p[$i]][]=$i;
		}
		$flag=false;
		for($i=0;$i<$this->k;$i++){
			$cluster=$this->clusters[$i];
			$m=$this->getMean($cluster);
			if($this->means[$i]!==$m){
				$this->means[$i]=$m;
				$flag=true;
			}
		}
		return $flag;
	}
	
	public function main(){
		$this->initMeans();
		while($this->setMeans()){
			
		}
	}
}
?>
