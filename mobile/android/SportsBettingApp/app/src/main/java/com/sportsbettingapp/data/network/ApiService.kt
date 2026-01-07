package com.sportsbettingapp.data.network

import com.sportsbettingapp.data.model.Bet
import com.sportsbettingapp.data.model.Prediction
import com.sportsbettingapp.data.model.RunRequest
import com.sportsbettingapp.data.model.RunResponse
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Path
import retrofit2.http.Query

interface ApiService {
    @POST("api/runs")
    suspend fun createRun(@Body request: RunRequest): RunResponse

    @GET("api/runs/{runId}/predictions")
    suspend fun getPredictions(@Path("runId") runId: String): List<Prediction>

    @GET("api/bets")
    suspend fun getBets(
        @Query("date") date: String,
        @Query("sport") sport: String
    ): List<Bet>
}
