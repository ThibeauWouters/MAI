package session6;

public class Programmer extends Employee {
	
	private String favouriteLanguage;
	private int linesWritten;
	private final double baseSalary = 10000;
	private Object getFavouriteLanguage;
		
	public Programmer(String name, String gender, int id, int linesWritten, String language) {
		super(name, gender, id);
		this.linesWritten = linesWritten;
		this.favouriteLanguage = language;
	}

	@Override
	public double calculateSalary() {
		double currentWage = this.baseSalary + this.getLinesWritten();
		if (this.getFavouriteLanguage.equals("Java")) {
			return 2*currentWage;
		} else {
			return currentWage;			
		}
	}
	
	// Getters and setters
	
	public void setFavouriteLanguage(String newLanguage) {
		this.favouriteLanguage = newLanguage;
	}
	
	public String getFavouriteLanguage() {
		return this.favouriteLanguage;
	}
	
	public void setLinesWritten(int newLines) {
		this.linesWritten = newLines;
	}
	
	public int getLinesWritten() {
		return this.linesWritten;
	}
}
