package session6;

public class SalesPerson extends Employee {
	
	private int itemsSold;
	private final double salaryFactor = 10;
	
	public SalesPerson(String name, String gender, int id, int itemsSold) {
		super(name, gender, id);
		this.itemsSold = itemsSold;
	}
	
	@Override
	public double calculateSalary() {
		return this.getItemsSold()*this.salaryFactor ;
	}
	
	// Getters and setters
	
	public int getItemsSold() {
		return this.itemsSold;
	}
}
