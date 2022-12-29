package Alignments;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

public abstract class Alignment {
	
	protected HashMap<String, String> alignment      = new HashMap<String, String>();
	public final static Set<Character> NUCLEOTIDES = new HashSet<>(Arrays.asList('A', 'C', 'T', 'G', '.')); 
}
