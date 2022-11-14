package session7;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.InputMismatchException;
import java.util.Scanner;
import java.util.StringTokenizer;

public class MuffinBakery {

	/* Variables */
	// Prices are in euro
	public static final double PRICE_BLUEBERRY = 3;
	public static final double PRICE_CHOCOLATE = 2.5;
	public static final double PRICE_REDVELVET = 3.5;
	
	private int numberOfBlueberry;
	private int numberOfChocolate;
	private int numberOfRedVelvet;
	
	/* Constructor */

	public MuffinBakery(String orderFileName) {
		this.numberOfBlueberry = 0;
		this.numberOfChocolate = 0;
		this.numberOfRedVelvet = 0;
		
		// Take the orders
		try(Scanner input = new Scanner(new FileReader(orderFileName));){
			// Keep going as long as there's input:
			while(input.hasNextLine()) {
				try {
					String line = input.nextLine();
					System.out.println(line);
					StringTokenizer tokenizer = new StringTokenizer(line);
					
					String name = tokenizer.nextToken(",").trim();
					int blue = Integer.parseInt(tokenizer.nextToken(",").trim());
					int choc = Integer.parseInt(tokenizer.nextToken(",").trim());
					int velvet = Integer.parseInt(tokenizer.nextToken(",").trim());
					
					// Update numbers for muffins
					this.setNumberOfBlueberry(this.numberOfBlueberry + blue);
					this.setNumberOfChocolate(this.numberOfChocolate + choc);
					this.setNumberOfRedVelvet(this.numberOfRedVelvet + velvet);
					
					// Alternative:
					
					//input.useDelimiter(Pattern.compile("(,)|(\r\n)"));
					//input.useLocale(Locale.ENGLISH);
					
					// Make the invoice
					makeInvoice(name, blue, choc, velvet);
					
					
				} catch (InputMismatchException ime) {
					System.out.println("Wrong input observed. Breaking.");
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
		} 
	}
	
	/* Other methods */
	
	private static void makeInvoice(String name, int bb, int choc, int velvet) {
		String invoiceFolderName = "src/";
		String fileName = invoiceFolderName + name + "_invoice.txt";
		System.out.println("Saving invoice for " + name + " to " + fileName);
		
		double price = calculatePrice(bb, choc, velvet);
		
		try(BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))){
			bw.write(name + " ordered " + bb + " bb muffins and " + choc + " choc and " + velvet + " red velvet. Price is " + price);
		}
		catch (FileNotFoundException fnf) {
			System.out.println("File not found when writing invoice " + fnf);
		} 
		catch (IOException ioe) {
			System.out.println("IOexception when writing invoice " + ioe);
		} 
		
	}
	
	public static double calculatePrice(int bb, int choc, int velvet) {
		return PRICE_BLUEBERRY*bb + PRICE_CHOCOLATE*choc + PRICE_REDVELVET*velvet;
	}
	
	/* Getters */
	public int getNumberOfBlueberry() {
		return numberOfBlueberry;
	}
	public int getNumberOfChocolate() {
		return numberOfChocolate;
	}
	public int getNumberOfRedVelvet() {
		return numberOfRedVelvet;
	}
	public int getTotalAmountOfMuffins() {
		return this.numberOfBlueberry + this.numberOfChocolate + this.numberOfRedVelvet;
	}
	
	/* Setters */

	public void setNumberOfBlueberry(int numberOfBlueberry) {
		this.numberOfBlueberry = numberOfBlueberry; 
	}
	public void setNumberOfChocolate(int numberOfChocolate) {
		this.numberOfChocolate = numberOfChocolate;
	}
	public void setNumberOfRedVelvet(int numberOfRedVelvet) {
		this.numberOfRedVelvet = numberOfRedVelvet;
	}

	public static void main(String[] args) {
		MuffinBakery mfb = new MuffinBakery("src/orders.txt");
	}











}
