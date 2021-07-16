#ifndef _IO_SPGEMM_H
#define _IO_SPGEMM_H

#include "Triple.h"
#include "CSC.h"
#include <fstream>
#include <sstream>

#define READBUFFER (512 * 1024 * 1024)  // in MB

template <typename IT, typename NT>
int ReadBinary(string filename, CSC<IT,NT> & csc)
{
    FILE * f = fopen(filename.c_str(), "r");
    if(!f)
    {
        cerr << "Problem reading binary input file" << filename << endl;
        return -1;
    }
    IT m,n,nnz;
    fread(&m, sizeof(IT), 1, f);
    fread(&n, sizeof(IT), 1, f);
    fread(&nnz, sizeof(IT), 1, f);
    
    if (m <= 0 || n <= 0 || nnz <= 0)
    {
        cerr << "Problem with matrix size in binary input file" << filename << endl;
        return -1;
    }
    double start = omp_get_wtime( );
    cout << "Reading matrix with dimensions: "<< m << "-by-" << n <<" having "<< nnz << " nonzeros" << endl;
    
    IT * rowindices = new IT[nnz];
    IT * colindices = new IT[nnz];
    NT * vals = new NT[nnz];
    
    size_t rows = fread(rowindices, sizeof(IT), nnz, f);
    size_t cols = fread(colindices, sizeof(IT), nnz, f);
    size_t nums = fread(vals, sizeof(NT), nnz, f);
    
    if(rows != nnz || cols != nnz || nums != nnz)
    {
        cerr << "Problem with FREAD, aborting... " << endl;
        return -1;
    }
    double end = omp_get_wtime( );
    // printf("start = %.16g\nend = %.16g\ndiff = %.16g\n", start, end, end - start);
    printf("Converting matrix data from binary to COO fromat takes %.16g seconds.\n", end - start);

    fclose(f);
    
    csc = *(new CSC<IT,NT>(rowindices, colindices, vals , nnz, m, n));
    
    delete [] rowindices;
    delete [] colindices;
    delete [] vals;
    return 1;
}

template <typename IT, typename NT>
int ReadASCII(string filename, CSC<IT,NT> & csc)
{
    bool isSymmetric = false;
    double start = omp_get_wtime( );
    ifstream infile(filename.c_str());
    char line[256];
    char c = infile.get();
    while(c == '%')
    {
        infile.getline(line,256);
        if (strstr(line, "symmetric")) {
            isSymmetric = true;
            cout << "Matrix is symmetric" << endl;
        }
        c = infile.get();
    }
    infile.unget();

    infile.getline(line,256);
    IT m,n,nnz;
    std::stringstream ss{line};
    ss >> m >> n >> nnz;
    
    if (isSymmetric) {
        nnz *= 2;
    }
    
    Triple<IT,NT> * triples = new Triple<IT,NT>[nnz];
    if (infile.is_open())
    {
        IT cnz = 0;	// current number of nonzeros
        while (! infile.eof() && cnz < nnz)
        {
            infile.getline(line,256);
            char *ch = line;
            triples[cnz].row = (IT)(atoi(ch));
            ch = strchr(ch, ' ');
            ch++;
            triples[cnz].col = (IT)(atoi(ch));
            ch = strchr(ch, ' ');
            if (ch != NULL) {
                ch++;
                /* Read third word (value data)*/
                triples[cnz].val = (NT)(atoi(ch));
                ch = strchr(ch, ' ');
            }
            else {
                triples[cnz].val = 1.0;
            }
            // infile >> triples[cnz].row >> triples[cnz].col >> triples[cnz].val;	// row-col-value
            triples[cnz].row--;
            triples[cnz].col--;
            if (isSymmetric) {
                if (triples[cnz].col != triples[cnz].row) {
                    cnz++;
                    triples[cnz].col = triples[cnz - 1].row;
                    triples[cnz].row = triples[cnz - 1].col;
                    triples[cnz].val = triples[cnz - 1].val;
                }
                else {
                    nnz--;
                }
            }
            ++cnz;
        }
        assert(cnz == nnz);
    }
    
    double end = omp_get_wtime( );
    // printf("start = %.16g\nend = %.16g\ndiff = %.16g\n", start, end, end - start);
    printf("Converting matrix data from ASCII to COO format: %.16g seconds\n", end - start);
    std::cout << "Input Matrix: Rows = " << m << ", Columns= " << n << ", nnz = " << nnz << std::endl;

    cout << "Converting to csc ... " << endl << endl;
    csc= *(new CSC<IT,NT>(triples, nnz, m, n));
    csc.totalcols = n;
    delete [] triples;
    return 1;
}



template <typename IT, typename NT>
int ReadASCII_Triples(string	  filename,
					  IT		 &nr,
					  IT		 &nc,
					  IT		 &nvals,
					  IT		*&rids,
					  IT		*&cids,
					  NT        *&vals,
					  bool		  remove_diags = false,
					  bool 		  rand_vals = false)
{
    bool isSymmetric = false;
    double start = omp_get_wtime( );
    ifstream infile(filename.c_str());
    char line[256];
    char c = infile.get();
    while(c == '%')
    {
        infile.getline(line,256);
        if (strstr(line, "symmetric")) {
            isSymmetric = true;
            cout << "Matrix is symmetric" << endl;
        }
        c = infile.get();
    }
    infile.unget();

	if (remove_diags)
		cout << "will remove diagonals from the matrix" << endl;

	IT m,n,nnz;
    std::stringstream ss{line};
    ss >> m >> n >> nnz;
    
    if (isSymmetric) {
        nnz *= 2;
    }

	rids = new IT[nnz];
	cids = new IT[nnz];
	vals = new NT[nnz];
    if (infile.is_open())
    {
        IT cnz = 0;
        while (! infile.eof() && cnz < nnz)
        {
            infile.getline(line,256);
            char *ch = line;
            IT rid = (IT)(atoi(ch));
            ch = strchr(ch, ' ');
            ch++;
            IT cid = (IT)(atoi(ch));
            ch = strchr(ch, ' ');
			NT val = 1.0;
			if (ch != NULL)
			{
                ch++;
                val = (NT)(atoi(ch));
                ch = strchr(ch, ' ');
			}

			if (rand_vals)
				val = rand()%10 + 7;
            
			if (remove_diags && rid == cid)
			{
				--nnz;
				if (isSymmetric)
					--nnz;
				continue;
			}

			rids[cnz] = rid-1;
			cids[cnz] = cid-1;
			vals[cnz] = val;
            if (isSymmetric)
			{
                if (rids[cnz] != cids[cnz])
				{
					cnz++;
					rids[cnz] = cids[cnz-1];
					cids[cnz] = rids[cnz-1];
					vals[cnz] = vals[cnz-1];
                }
                else
                    nnz--;
            }
            ++cnz;
        }
        assert(cnz == nnz);
    }
    
    double end = omp_get_wtime( );

	nr	  = m;
	nc	  = n;
	nvals = nnz;
	
    return 1;
}

#endif
