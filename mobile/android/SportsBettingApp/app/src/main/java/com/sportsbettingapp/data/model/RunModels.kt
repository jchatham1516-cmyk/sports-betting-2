package com.sportsbettingapp.data.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class RunRequestDto(
    val sport: String,
    @SerialName("game_date") val gameDate: String,
    val settings: Map<String, String> = emptyMap()
)

@Serializable
data class RunResponseDto(
    @SerialName("run_id") val runId: String,
    val status: String? = null,
    @SerialName("predictions_count") val predictionsCount: Int? = null,
    @SerialName("tracked_bets_count") val trackedBetsCount: Int? = null
)

@Serializable
data class RunStatusResponseDto(
    val id: String,
    val status: String,
    @SerialName("progress_percent") val progressPercent: Int,
    val message: String? = null,
    @SerialName("predictions_count") val predictionsCount: Int,
    @SerialName("tracked_bets_count") val trackedBetsCount: Int,
    val error: String? = null
)
