#ifndef SPA_H_
#define SPA_H_

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <algorithm>

#include "utility.h"


// SPA that is thread-safe but not multithreaded
template <class IT, class NT>
class SPA
{
public:
    SPA(IT range)
    {
        values.resize(range);
        flags.resize(range, 0);
    }
    
    template<typename KeyIterator>
    void Initialize(KeyIterator first, KeyIterator last)
    {
        while(first != last)
        {
            flags[*first] = 1;
            ++first;
        }
    }
    template <typename AddOperation>
    void Insert(IT tkey, NT tval, AddOperation addop)
    {
        if(flags[tkey] == 1)
        {
            values[tkey] = tval;
            flags[tkey] = 2;
            present.push_back(tkey);
        }
        else if(flags[tkey] == 2)   // previously set
        {
            values[tkey] = addop(tval, values[tkey]);
        }
    }
    size_t Size()
    {
        return present.size();
    }
    
    template<typename KeyIterator, typename ValueIterator>
    void OutputReset(KeyIterator firstkey, ValueIterator firstvalue)
    {
        // the range starting from first is large enough to hold all output elements
        for(auto index:present)
        {
            (*firstkey) = index;
            (*firstvalue) = values[index];
            ++firstkey;
            ++firstvalue;
            flags[index] = 0;
        }
        present.clear();
    }

    vector<NT> values;
    vector<short> flags;  // 0: not-allowed (masked out), 1: allowed, 2: set
    vector<IT> present;
};

// SPA that is thread-safe but not multithreaded
template <class IT>
class SPAStructure
{
public:
    SPAStructure(IT range): committed(0)
    {
        flags.resize(range, 0);
    }
    
    template<typename KeyIterator>
    void Initialize(KeyIterator first, KeyIterator last)
    {
        while(first != last)
        {
            flags[*first] = 1;
            possible.push_back(*first);
            ++first;
        }
    }
    void Insert(IT tkey)
    {
        if(flags[tkey] == 1)
        {
            flags[tkey] = 2;
            ++committed;
        }
    }
    size_t Size()
    {
        return committed;
    }
    
    void Reset()
    {
        for(auto index:possible)
        {
            flags[index] = 0;
        }
        possible.clear();
    }

    IT committed;
    vector<short> flags;  // 0: not-allowed (masked out), 1: allowed, 2: set
    vector<IT> possible;
};

#endif
