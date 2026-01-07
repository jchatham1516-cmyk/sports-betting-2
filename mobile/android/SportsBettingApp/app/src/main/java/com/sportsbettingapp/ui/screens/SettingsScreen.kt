package com.sportsbettingapp.ui.screens

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.sportsbettingapp.BuildConfig

@Composable
fun SettingsScreen(
    onDisclaimerClick: () -> Unit,
    contentPadding: PaddingValues
) {
    val context = LocalContext.current
    val packageInfo = context.packageManager.getPackageInfo(context.packageName, 0)
    val versionName = packageInfo.versionName

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(contentPadding)
            .padding(16.dp)
    ) {
        Text(text = "Settings", style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(16.dp))
        Text(text = "Base URL: ${BuildConfig.BASE_URL}")
        Text(text = "App version: $versionName")
        Spacer(modifier = Modifier.height(24.dp))
        Button(
            onClick = onDisclaimerClick,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("View Disclaimer")
        }
    }
}
