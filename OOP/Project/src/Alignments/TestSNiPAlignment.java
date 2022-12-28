package Alignments;

import java.util.ArrayList;

public class TestSNiPAlignment {
	public static void main(String[] args) {
		
		// Read the fasta file
		FastaContents hiv = new FastaContents("hiv.fasta");

		// Store them in the alignment
		StandardAlignment su = new StandardAlignment(hiv);
		SNiPAlignment sa = new SNiPAlignment(hiv, "2002.A.CD.02.KTB035");
		
		/* All tests come here */
		System.out.println("---------------------");
		System.out.println("Check the reference genome");
		System.out.println(sa.getReferenceGenome());
		System.out.println(su.getGenome("2002.A.CD.02.KTB035"));
		System.out.println(sa.getGenome("2002.A.CD.02.KTB035"));
		
		System.out.println("---------------------");
		System.out.println("Testing difference score. Before computing:");
		System.out.println(sa.getDifferenceScore(sa.getGenome("2002.A.CD.02.KTB035")));
		
		
		System.out.println("---------------------");
		System.out.println("What does first genome look like then?");
		System.out.println(sa.getGenome(sa.getIdentifiers().get(1)));
		
		System.out.println("---------------------");
		System.out.println("Testing search genome");
		ArrayList<String> test = sa.searchGenomes("TTTTTT");
		System.out.println(test);

		System.out.println("---------------------");
		System.out.println("Testing add genome");
		sa.addGenome("newID", "XXX");
		System.out.println(sa.getGenome("newID"));

		System.out.println("---------------------");
		System.out.println("Testing getting genome");
		String test2 = sa.getGenome("newID");
		System.out.println(test2);

		System.out.println("---------------------");
		System.out.println("Testing getting id");
		String testgenome = "XXX";
		System.out.println("Test genome: " + testgenome);
		String test3 = sa.getIdentifier(testgenome);
		System.out.println(test3);

		System.out.println("---------------------");
		System.out.println("Testing replacing genome");
		sa.replaceGenome("newID", "newerID", "ACTAATCGATCG");
		System.out.println(sa.getGenome("newerID"));
		System.out.println(sa.getGenome("newID"));

		System.out.println("---------------------");
		System.out.println("Testing replaceSequenceGenome");
		System.out.println(sa.getGenome("2002.A.CD.02.KTB035"));
		sa.replaceSequenceGenome("2002.A.CD.02.KTB035", "A", "T");
		System.out.println(sa.getGenome("2002.A.CD.02.KTB035"));

		System.out.println("---------------------");
		int size = sa.getSize();
		System.out.println("Testing remove genome: before removal " + size);
		sa.removeGenome("heyheyhey");
		size = sa.getSize();
		System.out.println("Removed heyheyhey: " + size);

		sa.removeGenome("newerID");
		size = sa.getSize();
		System.out.println("Removed newerID: " + size);
		test2 = sa.getGenome("newerID");
		System.out.println(test2);

		System.out.println("---------------------");
		System.out.println("Testing replaceSequenceGenome. Before:");
		System.out.println(sa.getGenome(sa.getIdentifiers().get(0)));
		sa.replaceSequenceGenome(sa.getIdentifiers().get(0), "TTT", "AAA");
		System.out.println("After:");
		System.out.println(sa.getGenome(sa.getIdentifiers().get(0)));

		System.out.println("---------------------");
		System.out.println("Testing replaceSequenceAlignment. Before:");
		System.out.println(sa.getGenome(sa.getIdentifiers().get(40)));
		sa.replaceSequenceAlignment("T", "A");
		System.out.println("After:");
		System.out.println(sa.getGenome(sa.getIdentifiers().get(40)));

	}

}
