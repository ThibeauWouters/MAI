package session7;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class LineCounter {

	public static void main(String[] args) {
		
		// Iterate over arguments
		// Alternative:
//		for (String fileName : args)
		for(int i=0; i <args.length;i++) {
			int count = 0;
			String fileName = args[i];
			System.out.println("File number " + i + " has name " + fileName);
			try(BufferedReader br = new BufferedReader(new FileReader(fileName))){
				String line = br.readLine();
				while(line != null) {
					count++;
					line = br.readLine();
				}
			} catch (FileNotFoundException fnf) {
				System.out.println("File not found exception when reading: " + fnf);
				continue;
			} catch (IOException ioe) {
				System.out.println("IO exception when reading: " + ioe);
				continue;
			} finally {
				System.out.println("Number of lines: " + count);
			}
		}

	}

}
