package Main;


import Alignments.*;
import Team.*;

public class Main {

	public static void main(String[] args) {
		/*
		 * Testing the code and assemblying everything together here.
		 */
		
		// TODO - specify path in the appropriate manner!
		
		// Read the fasta file
		FastaContents fasta = new FastaContents("hiv.fasta");
		
		// Create standard alignment from it
		StandardAlignment standard = new StandardAlignment(fasta);
		
		// Read the teams file
		Team team = new Team("team.txt", standard);
		
		// Do interesting stuff . . . 
		
		System.out.println("---------------");
		System.out.println("Testing alignments");
		System.out.println(standard.getDifferenceScore(standard.getGenome("2002.A.CD.02.KTB035")));
		System.out.println("---------------");
		System.out.println("Testing teams");
		team.listTeamMembers();

	}

}
