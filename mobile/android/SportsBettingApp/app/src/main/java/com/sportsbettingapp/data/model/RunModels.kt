package com.sportsbettingapp.data.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class RunRequest(
    val sport: String,
    @SerialName("game_date") val gameDate: String,
    val settings: Map<String, String> = emptyMap()
)

@Serializable
data class RunResponse(
    @SerialName("run_id") val runId: String? = null,
    val id: String? = null
)
