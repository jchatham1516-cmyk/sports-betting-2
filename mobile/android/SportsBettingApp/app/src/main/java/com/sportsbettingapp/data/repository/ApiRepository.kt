package com.sportsbettingapp.data.repository

import com.sportsbettingapp.data.model.Bet
import com.sportsbettingapp.data.model.PredictionItemDto
import com.sportsbettingapp.data.model.PredictionsResponseDto
import com.sportsbettingapp.data.model.RunRequestDto
import com.sportsbettingapp.data.model.RunResponseDto
import com.sportsbettingapp.data.remote.ApiService
import com.sportsbettingapp.data.remote.RetrofitClient

class ApiRepository(private val apiService: ApiService) {
    suspend fun createRun(request: RunRequestDto): RunResponseDto = apiService.createRun(request)

    suspend fun getPredictions(runId: String): PredictionsResponseDto {
        val items: List<PredictionItemDto> = apiService.getPredictions(runId)
        return PredictionsResponseDto(runId = runId, items = items)
    }

    suspend fun getBets(date: String, sport: String): List<Bet> = apiService.getBets(date, sport)

    companion object {
        fun create(): ApiRepository = ApiRepository(RetrofitClient.createService())
    }
}
