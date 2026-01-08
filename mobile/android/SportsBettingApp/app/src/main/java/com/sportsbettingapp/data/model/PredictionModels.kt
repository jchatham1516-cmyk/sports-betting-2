package com.sportsbettingapp.data.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class PredictionsResponseDto(
    @SerialName("run_id") val runId: String,
    val items: List<PredictionItemDto>
)

@Serializable
data class PredictionItemDto(
    val date: String? = null,
    val home: String? = null,
    val away: String? = null,
    val pick: String? = null,
    @SerialName("primary_recommendation") val primaryRecommendation: String? = null,
    val confidence: String? = null,
    val odds: String? = null,
    val edge: String? = null,
    val market: String? = null,
    val price: Double? = null,
    val units: Double? = null
)
