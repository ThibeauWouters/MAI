package session7;

import java.io.*; 
import java.util.*; 

/**
 * Compute the average from a file
 * @author thibe
 *
 */

public class IOCalculator {
	public static void main(String[] args) { 
		double mean = 0, number = 0, sum = 0, count = 0;

		System.out.println("Enter a file name"); 
		Scanner scan = new Scanner(System.in);
		String fileName = scan.next();
		// Make filename complete with src folder:
		fileName = "src/" + fileName;
		System.out.println("Digging into file: " + fileName);
		// Don't need scanner anymore
		scan.close(); 
		// Here is where we read stuff:

		try(Scanner input = new Scanner(new FileReader(fileName));){
			// Keep going as long as there's input:
			while(input.hasNext()) {
				try {
					number = input.nextDouble();
					count++;
					sum += number;
				} catch (InputMismatchException ime) {
					System.out.println("Wrong input observed. BReaking the loop.");
					break;
				}
			}
		}
		catch (FileNotFoundException fnfe) {
			System.out.println("Did not find file. Program exits.");
			System.exit(0);
		}
		catch (Error e) {
			System.out.println("Undefined error:" + e);
			System.exit(0);
		} finally {
			if(count == 0) {
				System.out.println("Did not observe numbers.");	
				System.exit(0);
			} else {
				mean = sum/count;
				System.out.format("Your numbers have an average of %f", mean);
			}
		}
		
		
		// Write the average to the same file
		FileOutputStream fout;
		try {
			fout = new FileOutputStream(fileName, true);
			PrintWriter out = new PrintWriter(fout, true);
			out.println("\n" + mean);
			out.close(); // note: can also put in try block, now needs to be close!
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
}

