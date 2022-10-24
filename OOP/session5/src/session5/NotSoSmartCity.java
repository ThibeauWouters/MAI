package session5;

public class NotSoSmartCity {
	// Variables
	private String[] inhabitantNames;
	private int nbInhabitants;
	private double totalAmountOfMoney;
	private String name;
	
	// Constructor
	public NotSoSmartCity(String name, double totalAmountOfMoney, String[] inhabitantNames){
		if (name == null) {
			name = "UNKNOWN";
		}
		this.name = name;
		if (inhabitantNames == null) {
			inhabitantNames = new String[] {};
		}
		this.inhabitantNames = inhabitantNames;
		this.totalAmountOfMoney = totalAmountOfMoney;
		this.nbInhabitants = inhabitantNames.length;
	}
	
	// Methods
	public void printCityNameForEachInhabitant(){
		for(int i = 0; i < this.inhabitantNames.length; i++){
			System.out.println(this.name);
		}
	}
	
	public double getAverageAmountOfMoney() {
		double averageMoney;
		if (nbInhabitants == 0) {
			System.out.println("There are no inhabitants.");
			return 0;
		} else {
			averageMoney = totalAmountOfMoney/nbInhabitants;
			return averageMoney;
		}
	}
	
	public static String babyNameGenerator(String beginning, int lettersToAdd){	
		if(lettersToAdd <= 0){
			return beginning;
		}
		else{
			return babyNameGenerator(beginning+getRandomLetter(), lettersToAdd -1);
		}
	}
	
	private static char getRandomLetter(){
		String alphabet = "abcdefghijklmnopqrstuvwxyz";
		return alphabet.charAt((int) (Math.random()*alphabet.length()));
	}
	
	public void printInhabitantNames(){
		for(int i=0; i < nbInhabitants; i++){
			System.out.println(inhabitantNames[i]);
		}
	}
	
	public static void main(String[] args){
		// ArrayIndexOutOfBoundsException thrown:
		
		String[] inh = {"Boris Johnson"};
		NotSoSmartCity city = new NotSoSmartCity("London", 1000, inh);
		city.printInhabitantNames();
		
		// NullPointerException thrown:
		
		NotSoSmartCity city2 = new NotSoSmartCity("DumboTown", 4000.5, null);
		city2.printInhabitantNames();
		
		// StackOverflowError
		String babyName = babyNameGenerator("Karel",-1);
		System.out.println(babyName);
		
		//Infinite loop 
		//NotSoSmartCity city3 = new NotSoSmartCity("DumboTown", -1, 4000.5, new String[]{"Ralph"});
		//city3.printCityNameForEachInhabitant();
		
		
	}
}