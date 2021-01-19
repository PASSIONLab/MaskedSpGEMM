/**
 * @file
 *	ht.h
 *
 * @author
 *
 * @date
 *
 * @brief
 *	Hash table variants utilized in SpGEMM
 *
 * @todo
 *
 * @note
 *	
 */

#pragma once

#include <unistd.h>

#include <functional>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

using std::vector;		using std::pair;	using std::tuple;	using std::get;
using std::conditional;	using std::is_same;



// used in masked index set membership (tuple version - three elements)
template <typename K,
		  typename V1 = void,
		  typename V2 = void,
		  typename T =
		  typename conditional<is_same<K, uint32_t>::value, int64_t,
							   typename conditional<is_same<K, uint16_t>::value,
							   int32_t, K>::type>::type>
struct
map_lp
{
	const T	scale_		  = 107;
	const T	default_size_ = 16;
	const T	ne_			  = -1;
	T		size_;
	vector<tuple<T, V1, V2>>	v_;
	


	map_lp (T sz)
	{
		size_ = default_size_;
		while (size_ < sz)
			size_ = size_ << 1;
		
		v_.resize(size_);		
		for (auto it = v_.begin(); it < v_.end(); ++it)
			get<0>(*it) = ne_;
	}



	bool
	insert (K k, V1 val1, V2 val2)
	{
		T	key = static_cast<T>(k);
		T	hv	= (key * scale_) & (size_ - 1); // beware

		while (get<0>(v_[hv]) != ne_ && get<0>(v_[hv]) != key)
			hv = (hv + 1) & (size_ - 1);

		if (get<0>(v_[hv]) == ne_)
		{
			get<0>(v_[hv]) = key;
			get<1>(v_[hv]) = val1;
			get<2>(v_[hv]) = val2;
			return true;
		}

		return false;
	}



	T
	find (K k)
	{
		T	key = static_cast<T>(k);
		T	hv	= (key * scale_) & (size_ - 1);

		while (get<0>(v_[hv]) != ne_ && get<0>(v_[hv]) != key)
			hv = (hv + 1) & (size_ - 1);

		if (get<0>(v_[hv]) == ne_)
			return -1;
		return hv;
	}



	V1 &
	get1 (T hv)
	{
		return get<1>(v_[hv]);
	}



	V2 &
	get2 (T hv)
	{
		return get<2>(v_[hv]);
	}



	// gather valid values
	template <typename IT,
			  typename NT>
	void
	gather (IT *idx_ptr, NT *val_ptr)
	{
		for (auto it = v_.begin(); it < v_.end(); ++it)
		{
			if (get<0>(*it) != ne_ && get<2>(*it))
			{
				*idx_ptr = get<0>(*it);
				*val_ptr = get<1>(*it);
				++idx_ptr;
				++val_ptr;		
			}
			get<0>(*it) = ne_;
		}
	}
};



// used in masked index set membership (pair version - two elements)
template <typename K,
		  typename V1,
		  typename T>
struct
map_lp<K, V1, void, T>
{
	const T	scale_		  = 107;
	const T	default_size_ = 16;
	const T	ne_			  = -1;
	T		size_;
	vector<pair<T, V1>>	v_;
	


	map_lp (T sz)
	{
		size_ = default_size_;
		while (size_ < sz)
			size_ = size_ << 1;
		
		v_.resize(size_);		
		for (auto it = v_.begin(); it < v_.end(); ++it)
			it->first  = ne_;
	}



	bool
	insert (K k, V1 val)
	{
		T	key = static_cast<T>(k);
		T	hv	= (key * scale_) & (size_ - 1);

		while (v_[hv].first != ne_ && v_[hv].first != key)
			hv = (hv + 1) & (size_ - 1);

		if (v_[hv].first == ne_)
		{
			v_[hv].first  = key;
			v_[hv].second = val;
			return true;
		}

		return false;
	}



	T
	find (K k)
	{
		T	key = static_cast<T>(k);
		T	hv	= (key * scale_) & (size_ - 1);

		while (v_[hv].first != ne_ && v_[hv].first != key)
			hv = (hv + 1) & (size_ - 1);

		if (v_[hv].first == ne_)
			return -1;
		return hv;
	}
	


	T
	find (K k, T *loc)
	{
		T	key = static_cast<T>(k);
		T	hv	= (key * scale_) & (size_ - 1);

		while (v_[hv].first != ne_ && v_[hv].first != key)
			hv = (hv + 1) & (size_ - 1);

		*loc = hv;
		if (v_[hv].first == ne_)
			return -1;
		return hv;
	}



	// direct access with the hv to vector
	V1 &
	operator[] (T hv)
	{
		return v_[hv].second;
	}

	void reset()
	{
		for (auto it = v_.begin(); it < v_.end(); ++it)
		{
			get<0>(*it) = ne_;			
		}

	}
};



// key-only version, one element
template <typename K,
		  typename T>
struct
map_lp<K, void, void, T>
{
	const T		scale_		  = 107;
	const T		default_size_ = 16;
	const T		ne_			  = -1;
	T			size_;
	vector<T>	v_;
	


	map_lp (T sz)
	{
		size_ = default_size_;
		while (size_ < sz)
			size_ = size_ << 1;
		
		v_.resize(size_);
		fill(v_.begin(), v_.end(), ne_);
	}



	bool
	insert (K k)
	{
		T	key = static_cast<T>(k);
		T	hv	= (key * scale_) & (size_ - 1);

		while (v_[hv] != ne_ && v_[hv] != key)
			hv = (hv + 1) & (size_ - 1);

		if (v_[hv] == ne_)
		{
			v_[hv] = key;
			return true;
		}

		return false;
	}



	T
	find (K k)
	{
		T	key = static_cast<T>(k);
		T	hv	= (key * scale_) & (size_ - 1);

		while (v_[hv] != ne_ && v_[hv] != key)
			hv = (hv + 1) & (size_ - 1);

		if (v_[hv] == ne_)
			return -1;
		return hv;
	}



	// direct access with the hv to vector
	K &
	operator[] (T hv)
	{
		return v_[hv];
	}
};	

