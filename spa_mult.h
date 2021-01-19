#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <algorithm>

#include "utility.h"
#include "CSR.h"
#include "SPA.h"


template <typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void SPASpGEMM(const CSR<IT, NT> & A, const CSR<IT, NT> & B, CSR<IT, NT> & C, CSR<IT, NT> & Mask,
               MultiplyOperation multop, AddOperation addop)
{
    C.rows = A.rows;
    C.cols = B.cols;
    C.zerobased = true;
    C.rowptr = my_malloc<IT>(C.rows + 1);
    IT * row_nz = my_malloc<IT>(C.rows);
    
    BIN<IT, NT> bin(A.rows, IMB_PWMIN);

    /* Set max bin */
    bin.set_max_bin(A.rowptr, A.colids, B.rowptr, C.rows, C.cols);

    /* Create hash table (thread local) */
    // bin.create_local_hash_table(c.cols);
    
    // Aydin Buluc: likely load-imbalanced way of parallelizing, improve later
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int threadnum = omp_get_thread_num();
        int numthreads = omp_get_num_threads();
        size_t low  = bin.rows_offset[tid];
        size_t high = bin.rows_offset[tid + 1];
        // size_t low = C.rows*threadnum/numthreads;
        // size_t high = C.rows*(threadnum+1)/numthreads;
        
        SPAStructure<IT> spastr(C.cols);
        for (size_t i=low; i<high; i++)
        {
            spastr.Initialize(Mask.colids + Mask.rowptr[i], Mask.colids + Mask.rowptr[i+1]);
            for(size_t j = A.rowptr[i]; j< A.rowptr[i+1]; ++j)
            {
                size_t rowofb = A.colids[j];
                for(size_t k = B.rowptr[rowofb]; k < B.rowptr[rowofb+1]; ++k)
                {
                    spastr.Insert(B.colids[k]);
                }
            }
            row_nz[i] = spastr.Size();
            spastr.Reset();
        }
    }
    scan(row_nz, C.rowptr, C.rows + 1);
    my_free<IT>(row_nz);

    C.nnz = C.rowptr[C.rows];
    
    C.colids = my_malloc<IT>(C.nnz);
    C.values = my_malloc<NT>(C.nnz);

    // Aydin Buluc, To-Do: repeated binary searches on row_nz to find load balanced low/high positions
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int threadnum = omp_get_thread_num();
        int numthreads = omp_get_num_threads();
        size_t low  = bin.rows_offset[tid];
        size_t high = bin.rows_offset[tid + 1];
        // size_t low = C.rows*threadnum/numthreads;
        // size_t high = C.rows*(threadnum+1)/numthreads;
        
        SPA<IT,NT> spa(C.cols);
        for (size_t i=low; i<high; i++)
        {
            spa.Initialize(Mask.colids + Mask.rowptr[i], Mask.colids + Mask.rowptr[i+1]);
            for(size_t j = A.rowptr[i]; j< A.rowptr[i+1]; ++j)
            {
                size_t rowofb = A.colids[j];
                for(size_t k = B.rowptr[rowofb]; k < B.rowptr[rowofb+1]; ++k)
                {
                    NT intproduct = multop(A.values[j], B.values[k]);   // could be avoided with a better interface
                    spa.Insert(B.colids[k], intproduct, addop);
                }
            }
            spa.OutputReset(C.colids + C.rowptr[i], C.values + C.rowptr[i]);
        }
    }
}

