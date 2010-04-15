import java.io.PrintWriter;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.File;

public class GetAllSubPackages {
    static PrintWriter writer;
    
    public static void main(String[] args) {
        
        if (args.length < 2) {
            System.out.println("First command line argument is the destination file");
            System.out.println("where the package names will be stored");
            System.out.println("Rest of the arguments are directory paths to the packages");
            System.out.println("Example:");
            System.out.println(
                "java GetAllSubPackages packages.txt rootdir1 rootdir2 rootdirN");
            return;
        }
        
        try {
            writer = new PrintWriter(new BufferedWriter(new FileWriter(args[0])));
            for (int i=1; i<args.length; i++) {
                File root = new File(args[i]);
                if (root.isDirectory()) {
                    writeDirs(root, root);    
                }
            }
            writer.close();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
    
    private static void writeDirs(File root, File dir) {
        String files[] = dir.list();
        boolean fileFound = true;
        for (int i=0; i<files.length; i++) {
            File file = new File(dir,files[i]);
            if (file.isDirectory()) {            
                writeDirs(root,file);            
            } else if (fileFound && (files[i].endsWith(".class")

                    || files[i].endsWith(".java"))) {
                fileFound = false;
                if (root.equals(dir)) {
                    //writer.println(".");   This was incorrect assumption about Javadoc
                } else {
    
writer.println(dir.getPath().substring(root.getPath().length()+1).replace(File.separatorChar,'.'));                
                }
            }
        }
    }
}
