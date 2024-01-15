for i=1:18
    if i < 10
        filename = sprintf("avgdataplane0%d.PIV.dat",i);
    else
        filename =sprintf("avgdataplane%d.PIV.dat",i);
    end
  
    T = readtable(filename);
    M = T{:,:};
    savename = sprintf("P%d.mat",i);
    save(savename,"M");
end