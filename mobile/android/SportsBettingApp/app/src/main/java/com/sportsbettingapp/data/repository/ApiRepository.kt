package com.sportsbettingapp.data.repository

import com.jakewharton.retrofit2.converter.kotlinx.serialization.asConverterFactory
import com.sportsbettingapp.BuildConfig
import com.sportsbettingapp.data.model.Bet
import com.sportsbettingapp.data.model.Prediction
import com.sportsbettingapp.data.model.RunRequest
import com.sportsbettingapp.data.model.RunResponse
import com.sportsbettingapp.data.network.ApiService
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit

class ApiRepository(private val apiService: ApiService) {
    suspend fun createRun(request: RunRequest): RunResponse = apiService.createRun(request)

    suspend fun getPredictions(runId: String): List<Prediction> = apiService.getPredictions(runId)

    suspend fun getBets(date: String, sport: String): List<Bet> = apiService.getBets(date, sport)

    companion object {
        fun create(): ApiRepository {
            val json = Json { ignoreUnknownKeys = true }
            val contentType = "application/json".toMediaType()

            val logging = HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BODY
            }

            val clientBuilder = OkHttpClient.Builder()
            if (BuildConfig.FLAVOR == "dev") {
                clientBuilder.addInterceptor(logging)
            }

            val retrofit = Retrofit.Builder()
                .baseUrl(BuildConfig.BASE_URL)
                .client(clientBuilder.build())
                .addConverterFactory(json.asConverterFactory(contentType))
                .build()

            return ApiRepository(retrofit.create(ApiService::class.java))
        }
    }
}
