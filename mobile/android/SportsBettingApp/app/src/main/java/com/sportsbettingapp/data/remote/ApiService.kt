package com.sportsbettingapp.data.remote

import com.sportsbettingapp.data.model.Bet
import com.sportsbettingapp.data.model.PredictionItemDto
import com.sportsbettingapp.data.model.RunRequestDto
import com.sportsbettingapp.data.model.RunResponseDto
import com.sportsbettingapp.data.model.RunStatusResponseDto
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Path
import retrofit2.http.Query

interface ApiService {
    @POST("api/runs")
    suspend fun createRun(@Body request: RunRequestDto): RunResponseDto

    @GET("api/runs/{runId}/predictions")
    suspend fun getPredictions(@Path("runId") runId: String): List<PredictionItemDto>

    @GET("api/runs/{runId}")
    suspend fun getRunStatus(@Path("runId") runId: String): RunStatusResponseDto

    @GET("api/bets")
    suspend fun getBets(
        @Query("date") date: String,
        @Query("sport") sport: String
    ): List<Bet>
}
