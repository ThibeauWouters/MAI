package session7;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Set;
import java.util.StringTokenizer;

public class WordCounter {
	
	public static void main(String[] args) {
		String fileName = "wordcounter.txt";
		// Store in a hashmap
		HashMap<String, Integer> map = new HashMap<String, Integer>();
		
		try(BufferedReader br = new BufferedReader(new FileReader(fileName))){
			String line = br.readLine();
			while(line != null) {
				// Get tokenizer for this line
				StringTokenizer tokenizer = new StringTokenizer(line);
				int numberOfTokens = tokenizer.countTokens();
				
				// Go over all tokens of this line
				for(int i = 0; i<numberOfTokens; i++) {
					String next = tokenizer.nextToken();
					// Check if already seen:
					if (map.containsKey(next)) {
						int current = map.remove(next);
						map.put(next, current+1);
					} else {
						map.put(next, 1);
					}
				}				
				// Read next line
				line = br.readLine();
			}
		} catch (FileNotFoundException fnf) {
			System.out.println("File not found: " + fnf);
		} catch (IOException ioe) {
			System.out.println("IO exception occurred: " + ioe);
		}
		
		Set<String> allKeys = map.keySet();
		for(String word : allKeys) {
			System.out.println(word + " has " + map.get(word) + " occurences");
		}
		
		// Output what has been stored:
		
	}
	

}
