#ifndef _CSR_H_
#define _CSR_H_

#include "CSC.h"
#include "Deleter.h"
#include "Triple.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
//#include <tbb/scalable_allocator.h>

#include <omp.h>

#include "grb_util.h"
#include "utility.h"


using namespace std;

template <class IT, class NT> class CSR {
public:
  CSR() : nnz(0), rows(0), cols(0), zerobased(true) {}
  CSR(IT mynnz, IT m, IT n) : nnz(mynnz), rows(m), cols(n), zerobased(true) {
    // Constructing empty Csc objects (size = 0) are allowed (why wouldn't
    // they?).
    assert(rows != 0);
    rowptr = my_malloc<IT>(rows + 1);
    if (nnz > 0) {
      colids = my_malloc<IT>(nnz);
      values = my_malloc<NT>(nnz);
    }
  }

  CSR(graph &G);
  CSR(string filename);
  CSR(const CSC<IT, NT> &csc); // CSC -> CSR conversion
  CSR(const CSR<IT, NT> &rhs); // copy constructor
  CSR(const CSC<IT, NT> &csc, const bool transpose);
  // CSR(const GrB_Matrix &A); // construct from GraphBLAS matrix
  CSR(GrB_Matrix *A, bool dup_mat = false);		// construct from GraphBLAS
												// matrix pointers
  CSR(GrB_Matrix A); // construct from GraphBLAS matrix (unpack)
  
  CSR<IT, NT> &operator=(const CSR<IT, NT> &rhs); // assignment operator
  bool operator==(const CSR<IT, NT> &rhs);        // ridefinizione ==
  void shuffleIds(); // Randomly permutating column indices
  void sortIds();    // Permutating column indices in ascending order

  void make_empty() {
    if (nnz > 0) {
		if (colids != NULL)
      		my_free<IT>(colids);
		if (values != NULL)
			my_free<NT>(values);
      nnz = 0;
    }
    if (rows > 0) {
		if (rowptr != NULL)
			my_free<IT>(rowptr);
      rows = 0;
    }
    cols = 0;
  }

  ~CSR() { make_empty(); }
  bool ConvertOneBased() {
    if (!zerobased) // already one-based
      return false;
    transform(rowptr, rowptr + rows + 1, rowptr,
              bind2nd(plus<IT>(), static_cast<IT>(1)));
    transform(colids, colids + nnz, colids,
              bind2nd(plus<IT>(), static_cast<IT>(1)));
    zerobased = false;
    return true;
  }
  bool ConvertZeroBased() {
    if (zerobased)
      return true;
    transform(rowptr, rowptr + rows + 1, rowptr,
              bind2nd(plus<IT>(), static_cast<IT>(-1)));
    transform(colids, colids + nnz, colids,
              bind2nd(plus<IT>(), static_cast<IT>(-1)));
    zerobased = true;
    return false;
  }
  bool isEmpty() { return (nnz == 0); }

  NT sumall()
  {
      IT sum = 0;
      #pragma omp parallel for reduction (+:sum)
      for(IT i=0; i<nnz; ++i )
      {
          sum += values[i];
      }
      return sum;
  }
  void Sorted();


  void get_grb_mat(GrB_Matrix *A);
  void get_grb_mat(GrB_Matrix A);
  void get_grb_mat_ptr(GrB_Matrix *A); // sets CSR object's pointers to NULL

  IT rows;
  IT cols;
  IT nnz; // number of nonzeros

  IT *rowptr;
  IT *colids;
  NT *values;
  bool zerobased;
};

// copy constructor
template <class IT, class NT>
CSR<IT, NT>::CSR(const CSR<IT, NT> &rhs)
    : nnz(rhs.nnz), rows(rhs.rows), cols(rhs.cols), zerobased(rhs.zerobased) {
  if (nnz > 0) {
    values = my_malloc<NT>(nnz);
    colids = my_malloc<IT>(nnz);
    copy(rhs.values, rhs.values + nnz, values);
    copy(rhs.colids, rhs.colids + nnz, colids);
  }
  if (rows > 0) {
    rowptr = my_malloc<IT>(rows + 1);
    copy(rhs.rowptr, rhs.rowptr + rows + 1, rowptr);
  }
}

template <class IT, class NT>
CSR<IT, NT> &CSR<IT, NT>::operator=(const CSR<IT, NT> &rhs) {
  if (this != &rhs) {
    if (nnz > 0) // if the existing object is not empty
    {
      my_free<IT>(colids);
      my_free<NT>(values);
    }
    if (rows > 0) {
      my_free<IT>(rowptr);
    }

    nnz = rhs.nnz;
    rows = rhs.rows;
    cols = rhs.cols;
    zerobased = rhs.zerobased;
    if (rhs.nnz > 0) // if the copied object is not empty
    {
      values = my_malloc<NT>(nnz);
      colids = my_malloc<IT>(nnz);
      copy(rhs.values, rhs.values + nnz, values);
      copy(rhs.colids, rhs.colids + nnz, colids);
    }
    if (rhs.cols > 0) {
      rowptr = my_malloc<IT>(rows + 1);
      copy(rhs.rowptr, rhs.rowptr + rows + 1, rowptr);
    }
  }
  return *this;
}

//! Construct a CSR object from a CSC
//! Accepts only zero based CSC inputs
template <class IT, class NT>
CSR<IT, NT>::CSR(const CSC<IT, NT> &csc)
    : nnz(csc.nnz), rows(csc.rows), cols(csc.cols), zerobased(true) {
  rowptr = my_malloc<IT>(rows + 1);
  colids = my_malloc<IT>(nnz);
  values = my_malloc<NT>(nnz);
  IT *work = my_malloc<IT>(rows);
  std::fill(work, work + rows, (IT)0); // initilized to zero
  for (IT k = 0; k < nnz; ++k) {
    IT tmp = csc.rowids[k];
    work[tmp]++; // row counts (i.e, w holds the "row difference array")
  }
IT last;

  if (nnz > 0) {
    rowptr[rows] = CumulativeSum(work, rows); // cumulative sum of w
    copy(work, work + rows, rowptr);

    for (IT i = 0; i < cols; ++i) {
      for (IT j = csc.colptr[i]; j < csc.colptr[i + 1]; ++j) {
        // last = work[csc.rowids[j]]++;
        // colids[last] = i;
        colids[last = work[csc.rowids[j]]++] = i;
        values[last] = csc.values[j];
      }
    }
  }
  my_free<IT>(work);
}

template <class IT, class NT>
CSR<IT, NT>::CSR(const CSC<IT, NT> &csc, const bool transpose)
    : nnz(csc.nnz), rows(csc.rows), cols(csc.cols), zerobased(true) {
  if (!transpose) {
    rowptr = my_malloc<IT>(rows + 1);
    colids = my_malloc<IT>(nnz);
    values = my_malloc<NT>(nnz);

    IT *work = my_malloc<IT>(rows);
    std::fill(work, work + rows, (IT)0); // initilized to zero

    for (IT k = 0; k < nnz; ++k) {
      IT tmp = csc.rowids[k];
      work[tmp]++; // row counts (i.e, w holds the "row difference array")
    }

    if (nnz > 0) {
      rowptr[rows] = CumulativeSum(work, rows); // cumulative sum of w
      copy(work, work + rows, rowptr);

      IT last;
      for (IT i = 0; i < cols; ++i) {
        for (IT j = csc.colptr[i]; j < csc.colptr[i + 1]; ++j) {
          colids[last = work[csc.rowids[j]]++] = i;
          values[last] = csc.values[j];
        }
      }
    }
    my_free<IT>(work);
  } else {
    rows = csc.cols;
    cols = csc.rows;
    rowptr = my_malloc<IT>(rows + 1);
    colids = my_malloc<IT>(nnz);
    values = my_malloc<NT>(nnz);

    for (IT k = 0; k < rows + 1; ++k) {
      rowptr[k] = csc.colptr[k];
    }
    for (IT k = 0; k < nnz; ++k) {
      values[k] = csc.values[k];
      colids[k] = csc.rowids[k];
    }
  }
}

template <class IT, class NT>
CSR<IT, NT>::CSR(graph &G) : nnz(G.m), rows(G.n), cols(G.n), zerobased(true) {
  // graph is like a triples object
  // typedef struct {
  // LONG_T m;
  // LONG_T n;
  // // Arrays of size 'm' storing the edge information
  // // A directed edge 'e' (0 <= e < m) from start[e] to end[e]
  // // had an integer weight w[e]
  // LONG_T* start;
  // LONG_T* end;
  // WEIGHT_T* w;
  // } graph;
  cout << "Graph nnz= " << G.m << " and n=" << G.n << endl;

  vector<Triple<IT, NT>> simpleG;
  vector<pair<pair<IT, IT>, NT>> currCol;
  currCol.push_back(make_pair(make_pair(G.start[0], G.end[0]), G.w[0]));
  for (IT k = 0; k < nnz - 1; ++k) {
    if (G.start[k] != G.start[k + 1]) {
      std::sort(currCol.begin(), currCol.end());
      simpleG.push_back(Triple<IT, NT>(
          currCol[0].first.first, currCol[0].first.second, currCol[0].second));
      for (int i = 0; i < currCol.size() - 1; ++i) {
        if (currCol[i].first == currCol[i + 1].first) {
          simpleG.back().val += currCol[i + 1].second;
        } else {
          simpleG.push_back(Triple<IT, NT>(currCol[i + 1].first.first,
                                           currCol[i + 1].first.second,
                                           currCol[i + 1].second));
        }
      }
      vector<pair<pair<IT, IT>, NT>>().swap(currCol);
    }
    currCol.push_back(
        make_pair(make_pair(G.start[k + 1], G.end[k + 1]), G.w[k + 1]));
  }

  // now do the last row
  sort(currCol.begin(), currCol.end());
  simpleG.push_back(Triple<IT, NT>(currCol[0].first.first,
                                   currCol[0].first.second, currCol[0].second));
  for (int i = 0; i < currCol.size() - 1; ++i) {
    if (currCol[i].first == currCol[i + 1].first) {
      simpleG.back().val += currCol[i + 1].second;
    } else {
      simpleG.push_back(Triple<IT, NT>(currCol[i + 1].first.first,
                                       currCol[i + 1].first.second,
                                       currCol[i + 1].second));
    }
  }

  nnz = simpleG.size();
  cout << "[After duplicate merging] Graph nnz= " << nnz << " and n=" << G.n
       << endl;

  rowptr = my_malloc<IT>(rows + 1);
  colids = my_malloc<IT>(nnz);
  values = my_malloc<NT>(nnz);

  IT *work = my_malloc<IT>(rows);
  std::fill(work, work + rows, (IT)0); // initilized to zero

  for (IT k = 0; k < nnz; ++k) {
    IT tmp = simpleG[k].row;
    work[tmp]++; // col counts (i.e, w holds the "col difference array")
  }

  if (nnz > 0) {
    rowptr[rows] = CumulativeSum(work, rows); // cumulative sum of w
    copy(work, work + rows, rowptr);

    IT last;
    for (IT k = 0; k < nnz; ++k) {
      colids[last = work[simpleG[k].row]++] = simpleG[k].col;
      values[last] = simpleG[k].val;
    }
  }
  my_free<IT>(work);
}



// template <class IT,
// 		  class NT>
// CSR<IT, NT>::CSR (const GrB_Matrix &A) :
// 	zerobased(true)
// {
// 	GrB_Index nc, nr, nv;
// 	GrB_Matrix_nrows(&nr, A);
// 	GrB_Matrix_ncols(&nc, A);
// 	GrB_Matrix_nvals(&nv, A);

// 	this->rows = static_cast<IT>(nr);
// 	this->cols = static_cast<IT>(nc);
// 	this->nnz  = static_cast<IT>(nv);

// 	// need cast from GrB_Index to IT
// 	GrB_Index	*rids = new GrB_Index[nv];
// 	GrB_Index	*cids = new GrB_Index[nv];
// 	this->rowptr      = my_malloc<IT>(this->rows+1);
// 	this->colids      = my_malloc<IT>(this->nnz);
// 	this->values	  = my_malloc<NT>(this->nnz);

// 	GrbMatrixExtractTuples<NT>()(rids, cids, this->values, &nv, A);
// 	assert(nv == this->nnz);

// 	// assume sorted and check it while forming
// 	memset(this->rowptr, 0, sizeof(IT) * (this->rows+1));
// 	GrB_Index last_rid = -1, last_cid = -1;
// 	for (GrB_Index i = 0; i < nv; ++i)
// 	{
// 		assert(rids[i] >= last_rid &&
// 			   "row ids are not sorted in the GraphBLAS matrix\n");
// 		if (rids[i] == last_rid)
// 			assert(cids[i] > last_cid &&
// 				   "col ids are not sorted in the GraphBLAS matrix\n");
// 		last_rid = rids[i];
// 		last_cid = cids[i];

// 		++this->rowptr[rids[i]+1];
// 		this->colids[i] = static_cast<IT>(cids[i]);
// 	}

// 	if (this->rows > 0)
// 		std::inclusive_scan(this->rowptr+1, this->rowptr+this->rows+1,
// 							this->rowptr+1);
	
// 	delete [] rids;
// 	delete [] cids;
// }



template <class IT,
		  class NT>
CSR<IT, NT>::CSR (GrB_Matrix A) :
	zerobased(true)
{
	static_assert(std::is_same<IT, GrB_Index>::value,
				  "CSR matrix index type and GrB_Matrix index type "
				  "must be the same");

	bool			is_iso, is_jumbled;
	GrB_Index		ap_size, aj_size, ax_size;
	GrB_Index		nr, nc, nnz;
	GrB_Descriptor	desc = NULL;

	GrB_Descriptor_new(&desc);
	
	GrB_Matrix_nrows(&nr, A);
	GrB_Matrix_ncols(&nc, A);
	GrB_Matrix_nvals(&nnz, A);
	this->rows = nr;
	this->cols = nc;
	this->nnz  = nnz;
	
	// does not free the matrix, but the matrix has no entries after this
	GxB_Matrix_unpack_CSR(A,
						  &this->rowptr,
						  &this->colids,
						  (void **)&this->values,
						  &ap_size,
						  &aj_size,
						  &ax_size,
						  &is_iso,
						  &is_jumbled,
						  desc);
	assert(!is_iso && "GraphBLAS matrix is iso-valued.");
	assert(!is_jumbled && "GraphBLAS matrix is not sorted\n");

    GrB_Descriptor_free(&desc);

	return;						  
}



template <class IT,
		  class NT>
CSR<IT, NT>::CSR (GrB_Matrix *A, bool dup_mat) :
	zerobased(true)
{
	static_assert(std::is_same<IT, GrB_Index>::value,
				  "CSR matrix index type and GrB_Matrix index type "
				  "must be the same");

	GrB_Matrix tmp = *A;			// shallow
	if (dup_mat)
		GrB_Matrix_dup(&tmp, *A); // deep

	GrB_Type		nz_type;
	bool			is_uniform, is_jumbled;
	GrB_Index		ap_size, aj_size, ax_size;
	GrB_Descriptor	desc = NULL;
	GrB_Descriptor_new(&desc);
	GxB_Matrix_type(&nz_type, tmp);	
	GrB_Matrix_nvals(&this->nnz, tmp);	
	GxB_Matrix_export_CSR(&tmp, &nz_type, &this->rows, &this->cols,
						  &this->rowptr, &this->colids, (void **)&this->values,
						  &ap_size, &aj_size, &ax_size,
						  &is_uniform, &is_jumbled,
						  desc); // frees the graphblas matrix
	assert(!is_jumbled && "GraphBLAS matrix is not sorted\n");


	return;						  
}





	

// check if sorted within rows?
template <class IT, class NT> void CSR<IT, NT>::Sorted() {
  bool sorted = true;
  for (IT i = 0; i < rows; ++i) {
    sorted &= my_is_sorted(colids + rowptr[i], colids + rowptr[i + 1],
                           std::less<IT>());
  }
    cout << "CSR graph is sorted by column id: "<< sorted << endl;
}

template <class IT, class NT>
bool CSR<IT, NT>::operator==(const CSR<IT, NT> &rhs) {
  bool same;
  if (nnz != rhs.nnz || rows != rhs.rows || cols != rhs.cols) {
    printf("%d:%d, %d:%d, %d:%d\n", nnz, rhs.nnz, rows, rhs.rows, cols,
           rhs.cols);
    return false;
  }
  if (zerobased != rhs.zerobased) {
    IT *tmp_rowptr = my_malloc<IT>(rows + 1);
    IT *tmp_colids = my_malloc<IT>(nnz);
    if (!zerobased) {
      for (int i = 0; i < rows + 1; ++i) {
        tmp_rowptr[i] = rowptr[i] - 1;
      }
      for (int i = 0; i < nnz; ++i) {
        tmp_colids[i] = colids[i] - 1;
      }
      same = std::equal(tmp_rowptr, tmp_rowptr + rows + 1, rhs.rowptr);
      same = same && std::equal(tmp_colids, tmp_colids + nnz, rhs.colids);
    } else if (!rhs.zerobased) {
      for (int i = 0; i < rows + 1; ++i) {
        tmp_rowptr[i] = rhs.rowptr[i] - 1;
      }
      for (int i = 0; i < nnz; ++i) {
        tmp_colids[i] = rhs.colids[i] - 1;
      }
      same = std::equal(tmp_rowptr, tmp_rowptr + rows + 1, rowptr);
      same = same && std::equal(tmp_colids, tmp_colids + nnz, colids);
    }
    my_free<IT>(tmp_rowptr);
    my_free<IT>(tmp_colids);
  } else {
    same = std::equal(rowptr, rowptr + rows + 1, rhs.rowptr);
    same = same && std::equal(colids, colids + nnz, rhs.colids);
  }

  bool samebefore = same;
  ErrorTolerantEqual<NT> epsilonequal(EPSILON);
  same = same && std::equal(values, values + nnz, rhs.values, epsilonequal);
  if (samebefore && (!same)) {
#ifdef DEBUG
    vector<NT> error(nnz);
    transform(values, values + nnz, rhs.values, error.begin(), absdiff<NT>());
    vector<pair<NT, NT>> error_original_pair(nnz);
    for (IT i = 0; i < nnz; ++i)
      error_original_pair[i] = make_pair(error[i], values[i]);
    if (error_original_pair.size() >
        10) { // otherwise would crush for small data
      partial_sort(error_original_pair.begin(),
                   error_original_pair.begin() + 10, error_original_pair.end(),
                   greater<pair<NT, NT>>());
      cout << "Highest 10 different entries are: " << endl;
      for (IT i = 0; i < 10; ++i)
        cout << "Diff: " << error_original_pair[i].first << " on "
             << error_original_pair[i].second << endl;
    } else {
      sort(error_original_pair.begin(), error_original_pair.end(),
           greater<pair<NT, NT>>());
      cout << "Highest different entries are: " << endl;
      for (typename vector<pair<NT, NT>>::iterator it =
               error_original_pair.begin();
           it != error_original_pair.end(); ++it)
        cout << "Diff: " << it->first << " on " << it->second << endl;
    }
#endif
  }
  return same;
}

template <class IT, class NT>
CSR<IT, NT>::CSR(const string filename) : zerobased(true) {
  IT i;
  bool isUnsy;
  IT num, offset, tmp_nz;
  char *line, *ch;
  FILE *fp;
  IT *col_coo, *row_coo;
  NT *val_coo;
  IT *each_row_index;
  IT *nnz_num;
  const int LINE_LENGTH_MAX = 256;

  isUnsy = false;
  line = (char *)malloc(sizeof(char) * LINE_LENGTH_MAX);

  /* Open File */
  fp = fopen(filename.c_str(), "r");
  if (fp == NULL) {
    exit(1);
  }
  do {
    fgets(line, LINE_LENGTH_MAX, fp);
    if (strstr(line, "general")) {
      isUnsy = true;
    }
  } while (line[0] == '%');

  /* Get size info */
  sscanf(line, "%d %d %d", &rows, &cols, &tmp_nz);

  /* Store in COO format */
  num = 0;
  col_coo = (IT *)malloc(sizeof(IT) * (tmp_nz));
  row_coo = (IT *)malloc(sizeof(IT) * (tmp_nz));
  val_coo = (NT *)malloc(sizeof(NT) * (tmp_nz));

  while (fgets(line, LINE_LENGTH_MAX, fp)) {
    ch = line;
    /* Read first word (row id)*/
    row_coo[num] = (IT)(atoi(ch) - 1);
    ch = strchr(ch, ' ');
    ch++;
    /* Read second word (column id)*/
    col_coo[num] = (IT)(atoi(ch) - 1);
    ch = strchr(ch, ' ');

    if (ch != NULL) {
      ch++;
      /* Read third word (value data)*/
      val_coo[num] = (NT)atof(ch);
      ch = strchr(ch, ' ');
    } else {
      val_coo[num] = 1.0;
    }
    num++;
  }
  fclose(fp);

  /* Count the number of non-zero in each row */
  nnz_num = (IT *)malloc(sizeof(IT) * rows);
  for (i = 0; i < rows; i++) {
    nnz_num[i] = 0;
  }
  for (i = 0; i < num; i++) {
    nnz_num[row_coo[i]]++;
    if (col_coo[i] != row_coo[i] && isUnsy == false) {
      nnz_num[col_coo[i]]++;
      (tmp_nz)++;
    }
  }

  nnz = tmp_nz;

  /* Allocation of rpt, col, val */
  rowptr = my_malloc<IT>(sizeof(IT) * (rows + 1));
  colids = my_malloc<IT>(sizeof(IT) * (nnz));
  values = my_malloc<NT>(sizeof(NT) * (nnz));

  offset = 0;
  for (i = 0; i < rows; i++) {
    rowptr[i] = offset;
    offset += nnz_num[i];
  }
  rowptr[rows] = offset;

  each_row_index = (IT *)malloc(sizeof(IT) * rows);
  for (i = 0; i < rows; i++) {
    each_row_index[i] = 0;
  }

  for (i = 0; i < num; i++) {
    colids[rowptr[row_coo[i]] + each_row_index[row_coo[i]]] = col_coo[i];
    values[rowptr[row_coo[i]] + each_row_index[row_coo[i]]++] = val_coo[i];

    if (col_coo[i] != row_coo[i] && isUnsy == false) {
      colids[rowptr[col_coo[i]] + each_row_index[col_coo[i]]] = row_coo[i];
      values[rowptr[col_coo[i]] + each_row_index[col_coo[i]]++] = val_coo[i];
    }
  }

  free(line);
  free(nnz_num);
  free(row_coo);
  free(col_coo);
  free(val_coo);
  free(each_row_index);
}

template <class IT, class NT> void CSR<IT, NT>::shuffleIds() {
  mt19937_64 mt(0);
  for (IT i = 0; i < rows; ++i) {
    IT offset = rowptr[i];
    IT width = rowptr[i + 1] - rowptr[i];
    uniform_int_distribution<IT> rand_scale(0, width - 1);
    for (IT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      IT target = rand_scale(mt);
      IT tmpId = colids[offset + target];
      NT tmpVal = values[offset + target];
      colids[offset + target] = colids[j];
      values[offset + target] = values[j];
      colids[j] = tmpId;
      values[j] = tmpVal;
    }
  }
}

template <class IT, class NT>
void CSR<IT,NT>::sortIds()
{
#pragma omp parallel for
    for (IT i = 0; i < rows; ++i)
    {
        vector< pair<IT,NT> > tosort;
        for (IT j = rowptr[i]; j < rowptr[i+1]; ++j)
        {
            tosort.push_back(make_pair(colids[j], values[j]));
        }
        std::sort(tosort.begin(), tosort.end());
        auto begitr = tosort.begin();
        for (IT j = rowptr[i]; j < rowptr[i+1]; ++j)
        {
            colids[j] = begitr->first;
            values[j] = begitr->second;
            ++begitr;
        }
    }
}


// A and B has to have sorted column ids
// Output will naturally have sorted ids
template <typename IT, typename NT, typename AddOperation>
CSR<IT,NT> Intersect(const CSR<IT,NT> & A, const CSR<IT,NT> & B, AddOperation addop)
{
    CSR<IT,NT> C;
    if (A.rows != B.rows || A.cols != B.cols) {
        std::cout << "Can not intersect due to dimension mismatch... "
        << A.rows << ":" << B.rows << ", " << A.cols << ":" << B.cols << std::endl;
      return C;
    }
    C.rows = A.rows;
    C.cols = A.cols;
    C.zerobased = A.zerobased;
    C.rowptr = my_malloc<IT>(C.rows + 1);
    IT * row_nz = my_malloc<IT>(C.rows);
    vector<vector<IT>> vec_colids(C.rows);
    vector<vector<NT>> vec_values(C.rows);
    
#pragma omp parallel for
    for(size_t i=0; i< A.rows; ++i)
    {
        IT acur = A.rowptr[i];
        IT aend = A.rowptr[i+1];
        IT bcur = B.rowptr[i];
        IT bend = B.rowptr[i+1];
        while(acur != aend && bcur != bend)
        {
            if(A.colids[acur] < B.colids[bcur]) ++acur;
            else if(A.colids[acur] > B.colids[bcur]) ++bcur;
            else    // they are equal
            {
                vec_colids[i].push_back(A.colids[acur]);
                vec_values[i].push_back(addop(A.values[acur], B.values[bcur]));
                ++acur; ++bcur;
            }
        }
        row_nz[i] = vec_colids[i].size();
    }

    scan(row_nz, C.rowptr, C.rows + 1);
    my_free<IT>(row_nz);
    
    C.nnz = C.rowptr[C.rows];
       
    C.colids = my_malloc<IT>(C.nnz);
    C.values = my_malloc<NT>(C.nnz);
#pragma omp parallel for
    for(size_t i=0; i< C.rows; ++i)
    {
        std::copy(vec_colids[i].begin(), vec_colids[i].end(), C.colids + C.rowptr[i]);
        std::copy(vec_values[i].begin(), vec_values[i].end(), C.values + C.rowptr[i]);
    }
    return C;
}



template <typename IT,
		  typename NT>
void
CSR<IT, NT>::get_grb_mat
(
  	GrB_Matrix *A
)
{
	GrB_Index	*rinds = new GrB_Index[this->nnz];
	GrB_Index	*cinds = new GrB_Index[this->nnz];
	GrB_Index	 i	   = 0;
	int			 decr  = 1 - this->zerobased;
	for (IT r = 0; r < this->rows; ++r)
	{
		for (IT cidx = this->rowptr[r]; cidx < this->rowptr[r+1]; ++cidx)
		{
			rinds[i]   = static_cast<GrB_Index>(r-decr);
			cinds[i++] = static_cast<GrB_Index>(this->colids[cidx]-decr);
		}
	}

	if (A != NULL)
	{
		GrB_Matrix_clear(*A);
		*A = NULL;
	}

	GrbMatrixBuild<NT>()(A, rinds, cinds, this->values,
						 this->rows, this->cols, this->nnz);

	GrB_Index nr, nc, nv;
	GrB_Matrix_nrows(&nr, *A);
	GrB_Matrix_ncols(&nc, *A);
	GrB_Matrix_nvals(&nv, *A);
	// cout << "GrB Matrix: " << nr << " " << nc << " " << nv << endl;

	delete [] rinds;
	delete [] cinds;

	
	return;
}



template <typename IT,
		  typename NT>
void
CSR<IT, NT>::get_grb_mat
(
    GrB_Matrix A
)
{
	assert(A != NULL && "GraphBLAS matrix to be packed is NULL!");
	
	GrB_Index nr, nc;
	GrB_Matrix_nrows(&nr, A);
	GrB_Matrix_ncols(&nc, A);

	assert(nr == this->rows && nc == this->cols &&
		   "Dimension mismatch in converting CSR matrix to GraphBLAS matrix.");

	bool is_iso = false, is_jumbled = false;
	GrB_Index ap_size = sizeof(IT) * (this->rows+1),
		aj_size = sizeof(IT) * this->nnz,
		ax_size = sizeof(NT) * this->nnz;

	GrB_Descriptor desc = NULL;
	GrB_Descriptor_new(&desc);
	
	GxB_Matrix_pack_CSR(A,
						&this->rowptr,
						&this->colids,
						(void **)&this->values,
						ap_size,
						aj_size,
						ax_size,
						is_iso,
						is_jumbled,
						desc);
	assert(this->rowptr == NULL && this->colids == NULL &&
		   this->values == NULL);

    GrB_Descriptor_free(&desc);
						
	return;					
}




template <typename IT,
		  typename NT>
void
CSR<IT, NT>::get_grb_mat_ptr
(
    GrB_Matrix *A
)
{
	static_assert(std::is_same<IT, GrB_Index>::value,
				  "CSR matrix index type and GrB_Matrix index type "
				  "must be the same");
	
	GrbAlgObj<NT> to_grb;
	GrB_Descriptor desc = NULL;
	// make sure CSR object is sorted
	GxB_Matrix_import_CSR(A, to_grb.get_type(), this->rows, this->cols,
						  &this->rowptr, &this->colids, (void **)&this->values,
						  sizeof(IT)*(this->rows+1), sizeof(NT)*this->nnz,
						  sizeof(NT)*this->nnz, false, false, desc);


	return;
}


#endif
