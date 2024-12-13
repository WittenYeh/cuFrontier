/*
 * @Author: Witten Yeh
 * @Date: 2024-10-02 21:19:47
 * @LastEditors: Witten Yeh
 * @LastEditTime: 2024-12-13 10:22:29
 * @Discription:
 */

#pragma once

#include <utility>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cstdlib> // for rand() and srand()
#include <ctime>   // for time()
#include <omp.h>   // for OpenMP
#include <cuda_runtime.h>

namespace cufrontier {
namespace implementation {

/**
 * @brief fill the dvector with index that set to be true in bool map
 * 
 * @param prefix_sum [in] -> prefix sum of bool map
 * @param boolmap [in] -> an array of bool variable
 * @param num_vertex [in] -> size of boolmap array
 * @param dvector [out] -> an empty dvector, which is alloced memory beforehand
 * @return none
 */
template <typename vertex_size_t>
__global__ void fill_dvector(
	const vertex_size_t* d_prefix_sum, 
	const bool *d_boolmap,
	const vertex_size_t num_vertex,
	vertex_size_t* d_active_vertex
) {
	using global_tid_t = vertex_size_t;
	global_tid_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (global_tid < num_vertex && d_boolmap[global_tid]) {
		d_active_vertex[d_prefix_sum[global_tid] - 1] = global_tid;
	}
}

template <typename vertex_size_t>
__global__ void gen_active_vertex(
	const bool* d_boolmap,
	const vertex_size_t num_vertex,
	vertex_size_t* d_active_vertex,
	vertex_size_t* d_num_active_vertex
) {
	using global_tid_t = vertex_size_t;
	global_tid_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (global_tid < num_vertex) {
		if (d_boolmap[global_tid]) {
			vertex_size_t local_wpos = atomicAdd(d_num_active_vertex, 1);
			d_active_vertex[local_wpos] = global_tid;
		}
	}
}

template <typename vertex_size_t>
void set_boolmap(
	bool* d_boolmap,
	const vertex_size_t num_vertex
) {
	using global_tid_t = vertex_size_t;
	global_tid_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (global_tid < num_vertex) {
		d_boolmap[global_tid] = true;
	}
}

template <typename vertex_size_t>
class Frontier {
	
	using vertex_id_t = vertex_size_t;

private:

	// * number of vertices
	vertex_size_t num_vertex;
	
	// * number of active vertex
	vertex_size_t num_active_vertex;
	
	// * internal bool map to record active vertex in the next iteration
	bool* d_boolmap;

	// * internal active vertices array
	vertex_id_t* d_active_vertex;

	const uint32_t kernel_block_size = 256;

	__host__ vertex_size_t gen_active_vertex_v1() {
		vertex_size_t* d_num_vertex;
		cudaMalloc(&d_num_vertex, sizeof(vertex_size_t));
		cudaMemset(d_num_vertex, 0, sizeof(vertex_size_t));
	
		const uint32_t num_block = (num_vertex + fill_kernel_block_size - 1) / fill_kernel_block_size;
		gen_active_vertex<vertex_size_t><<<num_block, kernel_block_size>>>(
			d_boolmap,
			num_vertex,
			d_active_vertex,
			d_num_vertex
		);
		// * set the number of active vertex
		vertex_size_t byte_selected	= 0;
		cudaMemcpy(&byte_selected, d_num_vertex, sizeof(vertex_size_t), cudaMemcpyDeviceToHost);
		return byte_selected;
	}

	__host__ vertex_size_t gen_active_vertex_v2() {
		vertex_size_t* d_prefix_sum;
		cudaMalloc(&d_prefix_sum, this->num_vertex * sizeof(vertex_size_t));

		// * use cub to compute prefix sum
		void* d_temp_storage = nullptr;
		std::size_t temp_storage_bytes = 0;
		// * to get temp storage requirement
		cub::DeviceScan::InclusiveSum(
			d_temp_storage,
			temp_storage_bytes, 
			d_boolmap,	 // d_in
			d_prefix_sum, // d_out
			num_vertex  // num_items
		);
		// alloc temp storage
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		// * invoke inclusive prefix sum function
		cub::DeviceScan::InclusiveSum(
			d_temp_storage, 
			temp_storage_bytes, 
			d_boolmap,	 // d_in
			d_prefix_sum, // d_out
			num_vertex  // num_items
		);

		// * call fill dvector
		const uint32_t num_block = (num_vertex + kernel_block_size - 1) / kernel_block_size;
		fill_dvector<vertex_size_t><<<num_block, kernel_block_size>>>(
			d_prefix_sum,
			d_boolmap,
			num_vertex,
			d_active_vertex
		);

		// * set the number of active vertex
		vertex_size_t byte_selected = 0;
		cudaMemcpy(
			&byte_selected, 
			d_prefix_sum + num_vertex - 1,
			sizeof(vertex_size_t),
			cudaMemcpyDeviceToHost
		);
	
		return byte_selected;
	}

public:

	__host__ __forceinline__
	Frontier(const vertex_size_t _num_vertex) : num_vertex(_num_vertex) {
		this->d_boolmap = static_cast<bool*>(cudaMalloc(sizeof(bool) * _num_vertex));

		/** 
		 * TODO: introduce low level virtual memory mamagement for high performance dynamic structure:
		 * https://developer.nvidia.com/blog/introducinory-management/  
		*/
		this->d_active_vertex = static_cast<vertex_id_t*>(cudaMalloc(sizeof(vertex_id_t) * _num_vertex));
	}

	__host__ __forceinline__
	~Frontier() {
		cudaFree(this->d_boolmap);
		cudaFree(this->d_active_vertex);
	}

	__host__ vertex_size_t get_num_vertex() const {
		return this->num_vertex;
	}

	__host__ __forceinline__
	bool* get_active_vertex() {
		return this->d_active_vertex;
	}

	__host__ __forceinline__
	void set_boolmap() {
		const uint32_t num_block = (num_vertex + kernel_block_size - 1) / kernel_block_size;
		set_boolmap<vertex_size_t><<<num_block, kernel_block_size>>>(
			this->d_boolmap,
			num_vertex
		);
	}

	template <bool use_atomic>
	__host__ __forceinline__
	void gen_active_vertex() {
		if constexpr (use_atomic) {
			this->num_active_vertex = this->gen_active_vertex_v1();
		} else {
			this->num_active_vertex = this->gen_active_vertex_v2();
		}
	}

};  // class Frontier

}  // namespace implementation
}  // namespace cufrontier
